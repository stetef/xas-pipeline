"""The recovery paths end to end, against a fake scheduler.

xas-rerun-orca, the shared :mod:`xas_pipeline.recovery` helpers, the rerun state
file, xas-rerun-corvus and the CORVUS auto-rerun triage, each driven in-process
with :func:`xas_pipeline.scheduler.run_command` (the single scheduler seam)
replaced by :class:`FakeScheduler`. Most tests pin one Phase 1 "Recovery" fix
from the 2026-10 clean-up (named in the test docstring); the watchdog's own
fixes are in tests/cli/test_watchdog.py.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from xas_pipeline import layout, recovery, remedy, rerun_state
from xas_pipeline import scheduler as _sched
from xas_pipeline.cli import auto_rerun_corvus, rerun_corvus, rerun_orca
from xas_pipeline.remedy import MAX_ATTEMPTS


# --------------------------------------------------------------------------- #
# fake scheduler (the run_command seam)
# --------------------------------------------------------------------------- #
class FakeScheduler:
    """Answers sbatch/scancel/squeue (and qsub/qdel/qstat) in memory.

    ``states`` maps job id -> RUNNING/PENDING; a submitted job starts PENDING and
    a cancelled one disappears. ``submits`` records each accepted submission.
    """

    def __init__(self):
        self.states: dict[str, str] = {}
        self.calls: list[list[str]] = []
        self.submits: list[dict] = []
        self.cancelled: list[str] = []
        self.fail_cancel: set[str] = set()
        self.fail_submit: set[str] = set()     # script names refused
        self._next = 1000

    @staticmethod
    def _done(stdout="", rc=0):
        return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="" if rc == 0 else "refused")

    def __call__(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        prog = cmd[0]
        if prog in ("sbatch", "qsub"):
            script = cmd[-1]
            if script in self.fail_submit:
                return self._done(rc=1)
            self._next += 1
            job_id = str(self._next)
            deps, extra = {}, []
            args = cmd[1:-1]
            i = 0
            while i < len(args):
                arg = args[i]
                if arg.startswith("--dependency="):
                    kind, _, ids = arg.removeprefix("--dependency=").partition(":")
                    deps[kind] = ids.split(":")
                elif arg == "-W":
                    kind, _, ids = args[i + 1].removeprefix("depend=").partition(":")
                    deps[kind] = ids.split(":")
                    i += 1
                else:
                    extra.append(arg)
                i += 1
            self.submits.append({"script": script, "cwd": Path(cwd) if cwd else None, "id": job_id,
                                 "afterok": deps.get("afterok", []), "afterany": deps.get("afterany", []),
                                 "extra": extra})
            self.states[job_id] = "PENDING"
            return self._done(f"Submitted batch job {job_id}\n" if prog == "sbatch" else f"{job_id}.pbs\n")
        if prog in ("scancel", "qdel"):
            job_id = cmd[1]
            if job_id in self.fail_cancel:
                return self._done(rc=1)
            self.cancelled.append(job_id)
            self.states.pop(job_id, None)
            return self._done()
        if prog == "squeue":
            ids = cmd[cmd.index("-j") + 1].split(",")
            with_state = "%i %T" in cmd
            lines = [f"{j} {self.states[j]}" if with_state else j for j in ids if j in self.states]
            return self._done("\n".join(lines) + ("\n" if lines else ""))
        if prog == "qstat":
            job_id = cmd[-1]
            letter = {"RUNNING": "R", "PENDING": "Q"}.get(self.states.get(job_id, ""))
            return self._done(f"job_state = {letter}\n") if letter else self._done(rc=153)
        raise AssertionError(f"unexpected scheduler command {cmd}")

    def by_script(self, suffix: str) -> list[dict]:
        return [s for s in self.submits if s["script"].endswith(suffix)]


@pytest.fixture
def fake(monkeypatch):
    sched = FakeScheduler()
    monkeypatch.setattr(_sched, "run_command", sched)
    return sched


# --------------------------------------------------------------------------- #
# failed ORCA run fixtures
# --------------------------------------------------------------------------- #
BASE_IN = textwrap.dedent(
    """\
    ! TightOPT PBE0 D3BJ RIJCOSX def2-TZVP def2/J CPCM(water)

    ! AnFreq

    %maxcore 1800

    %pal nprocs 16
      end

    *xyzfile 0 1 /abs/path/RUN/RUN_clean.xyz
    """
)

OOM_LOG = "\n".join(
    ["GEOMETRY OPTIMIZATION CYCLE"] * 3
    + ["No memory left for COSX RHS", "ORCA finished by error termination in NUMFREQ"]
)

SCF_LOG = "\n".join(
    ["GEOMETRY OPTIMIZATION CYCLE"] * 4
    + ["Warning: op=0 Small HOMO/LUMO gap ( -0.010) - skipping pre-diagonalization"]
    + [f"  {i + 1:>3}      {-100.0 + 1e-7 * (i % 3):.12f}     6.4e-06   (NR   MAcro)" for i in range(12)]
    + ["Error (ORCA_LEANSCF): unfortunately, the SCF has not converged"]
)


def _make_failed_run(parent: Path, run_id: str, log_body: str = OOM_LOG, *,
                     job_script: str = "#!/bin/bash\n#SBATCH --mem=16G\n",
                     batch_log: str | None = "job_name\tstatus\tjob_id\n") -> Path:
    """A failed run dir at ``parent/<run_id>``; batch-jobs.log written at *parent*."""
    run = parent / run_id
    run.mkdir(parents=True)
    (run / f"{run_id}-orca.log").write_text(log_body, encoding="utf-8")
    (run / f"{run_id}.gbw").write_bytes(b"ORBITALS")
    (run / f"{run_id}.xyz").write_text("1\nlast\nZn 0 0 0\n", encoding="utf-8")
    (run / f"{run_id}.in").write_text(BASE_IN.replace("RUN", run_id), encoding="utf-8")
    (run / f"generated-{run_id}-orca.script").write_text(job_script, encoding="utf-8")
    if batch_log is not None:
        (parent / "batch-jobs.log").write_text(batch_log, encoding="utf-8")
    return run


def _rerun(run: Path, scheduler: str = "slurm", **kw) -> str:
    return rerun_orca.rerun(run, scheduler=scheduler, **kw)


# --------------------------------------------------------------------------- #
# xas-rerun-orca (ported from test_auto_rerun.py)
# --------------------------------------------------------------------------- #
def test_rerun_orca_regates_a_pending_postprocess_onto_the_new_corvus_job(tmp_path, fake):
    """Fix: cancelling the stale CORVUS job could satisfy the postprocess's afterany,
    so it ran before the recomputed run; the watchdog re-gated, xas-rerun-orca did not."""
    run_id = "RG.S2N2_zn1"
    batch = tmp_path
    run = _make_failed_run(batch, run_id, batch_log=(
        "job_name\tstatus\tjob_id\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=700\n"
        f"postprocess-{batch.name}\tSUBMITTED\tjob_id=701\n"
    ))
    (batch / f"generated-postprocess-{batch.name}.script").write_text("#!/bin/bash\n", encoding="utf-8")
    fake.states.update({"700": "PENDING", "701": "PENDING"})

    assert _rerun(run) == "resubmitted"

    [post] = fake.by_script(f"generated-postprocess-{batch.name}.script")
    [corvus] = [s for s in fake.submits if "corvus" in s["script"]]
    assert corvus["id"] in post["afterany"]
    assert "701" in fake.cancelled and "700" in fake.cancelled


def test_attempts_do_not_stack_cards(tmp_path, fake):
    """Each remedy is applied to the pristine original, never to the last remedied
    input. Was a --no-submit subprocess loop; --dry-run now changes nothing
    (77e7365), so the two attempts really submit, to the fake scheduler."""
    run_id = "STK.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    for _ in range(2):  # two attempts on the same (unchanged) failing log
        assert _rerun(run) == "resubmitted"

    remedied = (run / f"{run_id}.in").read_text()
    assert remedied.count("%scf") == 0
    assert remedied.count(remedy.REMEDY_MARKER) == 1
    # Attempt 2's maxcore bump is x2.5 from the pristine 1800 (not stacked on x1.6).
    assert "%maxcore 4500" in remedied
    assert remedied.count("! MOREAD") == 1
    assert "! SlowConv" not in remedied and "Shift Shift" not in remedied
    # The job's memory request also scales from the pristine script: 16G x 2.5.
    assert "--mem=40G" in (run / f"generated-{run_id}-orca.script").read_text()
    history = run / f"{run_id}-rerun-history"
    assert (history / f"original-{run_id}.in").is_file()
    assert (history / "attempt1-STK.S2N2_zn1.in").is_file() and (history / "attempt2-STK.S2N2_zn1.in").is_file()
    state = json.loads((run / f"{run_id}-rerun-state.json").read_text())
    assert [a["remedy"] for a in state["attempts"]] == ["oom-maxcore-x1.6", "oom-maxcore-x2.5"]
    assert state["attempts"][1]["note"] == "mem 16G -> 40G"


def test_resubmits_orca_then_corvus_afterok(tmp_path, fake):
    run_id = "SUB.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    assert _rerun(run) == "resubmitted"
    [orca] = fake.by_script("-orca.script")
    [corvus] = fake.by_script("-corvus-xas-wrapper.script")
    assert orca["cwd"] == run and corvus["cwd"] == run
    assert corvus["afterok"] == [orca["id"]]
    log_text = (tmp_path / "batch-jobs.log").read_text()
    assert f"orca-rerun1-{run_id}\tSUBMITTED\tjob_id={orca['id']}" in log_text
    assert f"corvus-xas-rerun1-{run_id}\tSUBMITTED\tjob_id={corvus['id']}" in log_text
    state = json.loads((run / f"{run_id}-rerun-state.json").read_text())
    assert state["attempts"][0]["orca_job_id"] == orca["id"]
    assert state["attempts"][0]["corvus_job_id"] == corvus["id"]


def test_cancels_dependent_corvus_on_orca_failure(tmp_path, fake):
    """On an ORCA failure, the dependent CORVUS job is looked up and cancelled
    (here a non-remediable SCF failure: it is escalated as well)."""
    run_id = "CAN.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id, SCF_LOG, batch_log=(
        "job_name\tstatus\tjob_id\n"
        f"orca-{run_id}\tSUBMITTED\tjob_id=500\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n"
    ))
    fake.states.update({"501": "PENDING"})
    assert _rerun(run) == "escalated"
    assert fake.cancelled == ["501"]
    assert fake.submits == []
    log_text = (tmp_path / "batch-jobs.log").read_text()
    assert f"corvus-{run_id}\tCANCELLED" in log_text and "NEEDS_HUMAN" in log_text


def test_escalates_to_canonical_channels_not_txt(tmp_path, fake):
    """Exhausting the ladder records NEEDS_HUMAN in state + batch-jobs.log, no sidecar .txt."""
    run_id = "ESC.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    outcomes = [_rerun(run) for _ in range(MAX_ATTEMPTS + 1)]  # two remedied attempts, then escalation
    assert outcomes == ["resubmitted"] * MAX_ATTEMPTS + ["escalated"]

    assert not (run / f"{run_id}-needs-human.txt").exists()
    state = json.loads((run / f"{run_id}-rerun-state.json").read_text())
    assert state["resolution"] == "needs_human"
    assert len(state["attempts"]) == MAX_ATTEMPTS
    assert "NEEDS_HUMAN" in (tmp_path / "batch-jobs.log").read_text()
    # A fourth firing of the hook is a no-op.
    n_calls = len(fake.calls)
    assert _rerun(run) == "terminal"
    assert len(fake.calls) == n_calls


def test_rechain_cancels_the_dead_stage_and_requeues_it_on_the_new_job(tmp_path, fake):
    """The scheduler killed the dependent --pre stage when its parent failed; the
    rerun has to put it back, or a recoverable failure silently drops it."""
    run_dir = tmp_path / "id-quick-ca-fixed"
    next_dir = tmp_path / "id-caopt-anfreq"
    run_dir.mkdir()
    next_dir.mkdir()
    (run_dir / "generated-id-quick-ca-fixed-orca.script").write_text("#!/bin/bash\n")
    (next_dir / "generated-id-caopt-anfreq-orca.script").write_text("#!/bin/bash\n")
    log = tmp_path / "batch-jobs.log"
    log.write_text("job_name\tstatus\tjob_id\n")
    record = {
        "next_run_id": "id-caopt-anfreq",
        "next_run_dir": str(next_dir),
        "next_job_script": "generated-id-caopt-anfreq-orca.script",
        "next_job_id": "555",
    }
    sidecar = run_dir / "id-quick-ca-fixed-next-stage.json"
    sidecar.write_text(json.dumps(record), encoding="utf-8")

    orca_id, follower = recovery.resubmit_orca(
        _sched.get_scheduler("slurm"), run_dir, "id-quick-ca-fixed", log, label="rerun1"
    )

    assert fake.cancelled == ["555"]
    assert [(s["script"], s["cwd"], s["afterok"]) for s in fake.submits] == [
        ("generated-id-quick-ca-fixed-orca.script", run_dir, []),
        ("generated-id-caopt-anfreq-orca.script", next_dir, [orca_id]),
    ]
    assert not fake.by_script("wrapper.script")  # a --pre stage has no CORVUS of its own
    # The sidecar now names the live job, so a second failure cancels the right one.
    assert json.loads(sidecar.read_text())["next_job_id"] == follower
    assert "orca-stage-rerun1-id-caopt-anfreq\tSUBMITTED" in log.read_text()


# --------------------------------------------------------------------------- #
# Phase 1 "Recovery" regressions
# --------------------------------------------------------------------------- #
def test_batch_log_is_found_by_walking_up_from_a_grouped_run(tmp_path, fake):
    """Fix: rerun_orca read batch-jobs.log from the run's parent (the group dir
    in a grouped batch), found none, and never cancelled the stale CORVUS."""
    batch = tmp_path / "batch"
    run_id = "4j1v_cluster2_Zn-caopt-anfreq"
    run = _make_failed_run(batch / "4j1v_cluster2_Zn", run_id, SCF_LOG, batch_log=None)
    (batch / "batch-jobs.log").write_text(
        "job_name\tstatus\tjob_id\n"
        f"orca-{run_id}\tSUBMITTED\tjob_id=500\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n",
        encoding="utf-8",
    )
    assert not (run.parent / "batch-jobs.log").exists()
    _rerun(run)
    assert fake.cancelled == ["501"]
    log_text = (batch / "batch-jobs.log").read_text()
    assert "CANCELLED" in log_text and "NEEDS_HUMAN" in log_text


def test_stale_corvus_is_matched_by_exact_run_id(tmp_path, fake):
    """Fix: suffix/substring run-id matching. A_zn1's failure must not cancel the
    CORVUS job of A_zn1-caopt-anfreq (or of an id ending in A_zn1)."""
    run = _make_failed_run(tmp_path, "A_zn1", SCF_LOG, batch_log=(
        "job_name\tstatus\tjob_id\n"
        "corvus-xas-A_zn1\tSUBMITTED\tjob_id=501\n"
        "corvus-xas-A_zn1-caopt-anfreq\tSUBMITTED\tjob_id=601\n"
        "corvus-xas-XA_zn1\tSUBMITTED\tjob_id=701\n"
    ))
    _rerun(run)
    assert fake.cancelled == ["501"]


def test_failed_cancel_of_stale_corvus_is_recorded(tmp_path, fake):
    """Fix: the cancel result was ignored; the outcome line now says it failed."""
    run_id = "CF.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id, SCF_LOG, batch_log=(
        "job_name\tstatus\tjob_id\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n"
    ))
    fake.fail_cancel.add("501")
    _rerun(run)
    assert "cancelled stale CORVUS job 501 (cancel failed)" in (tmp_path / "batch-jobs.log").read_text()


def test_moread_reads_a_copy_not_the_gbw_the_job_writes(tmp_path, fake):
    """Fix: %moinp pointed at <id>.gbw, the file the resubmitted job overwrites as
    it starts; the guess is now a copy, <id>-moread.gbw."""
    run_id = "MOR.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    assert _rerun(run) == "resubmitted"
    text = (run / f"{run_id}.in").read_text()
    assert f'%moinp "{run_id}-moread.gbw"' in text
    assert f'%moinp "{run_id}.gbw"' not in text
    assert (run / f"{run_id}-moread.gbw").read_bytes() == b"ORBITALS"
    assert (run / f"{run_id}.gbw").read_bytes() == b"ORBITALS"


def test_pbs_oom_remedy_scales_the_mem_request(tmp_path, fake):
    """Fix: the OOM remedy only knew Slurm's --mem; PBS's ``-l mem=`` stayed put."""
    run_id = "PBS.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id, job_script="#!/bin/bash\n#PBS -l mem=16gb\n#PBS -l ncpus=16\n")
    assert _rerun(run, scheduler="pbs") == "resubmitted"
    script = (run / f"generated-{run_id}-orca.script").read_text()
    assert "#PBS -l mem=26gb" in script  # 16 x 1.6 = 25.6 -> 26
    assert [c[0] for c in fake.calls if c[0] in ("qsub", "sbatch")] == ["qsub", "qsub"]
    [orca] = fake.by_script("-orca.script")
    [corvus] = fake.by_script("-corvus-xas-wrapper.script")
    assert corvus["afterok"] == [orca["id"]]


def test_dry_run_has_no_side_effects(tmp_path, fake):
    """Fix: --dry-run edited the input, wrote history and bumped the counter."""
    run_id = "DRY.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id, batch_log=(
        "job_name\tstatus\tjob_id\n"
        f"corvus-xas-{run_id}\tSUBMITTED\tjob_id=501\n"
    ))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    assert _rerun(run, submit=False) == "preview"

    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert fake.calls == []


def test_dry_run_cli_flag_has_no_side_effects(tmp_path, fake, monkeypatch):
    run_id = "DRYCLI.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    before = sorted(p.name for p in run.iterdir())
    monkeypatch.setattr("sys.argv", ["xas-rerun-orca", str(run), "--scheduler", "slurm", "--dry-run"])
    assert rerun_orca.main() == 0
    assert sorted(p.name for p in run.iterdir()) == before
    assert fake.calls == []


def test_submit_failure_is_recorded_and_escalated(tmp_path, fake):
    """Fix: a refused resubmission (after the stale CORVUS was cancelled) lost the
    run. The attempt is kept, the ladder ends, and the log says why."""
    run_id = "REF.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    fake.fail_submit.add(f"generated-{run_id}-orca.script")
    assert _rerun(run) == "escalated"
    state = json.loads((run / f"{run_id}-rerun-state.json").read_text())
    assert state["resolution"] == "needs_human"
    assert len(state["attempts"]) == 1 and state["attempts"][0]["orca_job_id"] is None
    log_text = (tmp_path / "batch-jobs.log").read_text()
    assert f"orca-rerun1-{run_id}\tSUBMIT_FAILED" in log_text
    assert "resubmission failed at attempt 1" in log_text
    # And the hook firing again does not retry it.
    assert _rerun(run) == "terminal"


def test_no_batch_log_means_no_resubmission(tmp_path, fake):
    run = _make_failed_run(tmp_path, "NOLOG.S2N2_zn1", batch_log=None)
    assert _rerun(run) == "error"
    assert fake.submits == []
    assert (run / "NOLOG.S2N2_zn1.in").read_text() == BASE_IN.replace("RUN", "NOLOG.S2N2_zn1")


# --------------------------------------------------------------------------- #
# rerun state: atomic writes, corrupt file -> needs_human
# --------------------------------------------------------------------------- #
def test_state_round_trips_and_leaves_no_temp_file(tmp_path):
    path = rerun_state.state_path(tmp_path, "X_zn1", kind="corvus")
    state = rerun_state.RerunState("X_zn1", [rerun_state.Attempt(1, "k", "r", "t", corvus_job_id="9")])
    rerun_state.save_state(path, state)
    assert rerun_state.load_state(path, "X_zn1") == state
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_state_write_is_atomic(tmp_path, monkeypatch):
    """Fix: a job killed mid-write truncated the state (and so reset the ladder).
    The new state is written aside and swapped in; a failed swap leaves the old one."""
    path = rerun_state.state_path(tmp_path, "X_zn1")
    old = rerun_state.RerunState("X_zn1", [rerun_state.Attempt(1, "oom", "r", "t")])
    rerun_state.save_state(path, old)
    original = path.read_text()

    def killed(src, dst):
        raise OSError("killed mid-write")

    monkeypatch.setattr(rerun_state.os, "replace", killed)
    with pytest.raises(OSError):
        rerun_state.save_state(path, rerun_state.RerunState("X_zn1", old.attempts * 2))
    assert path.read_text() == original


@pytest.mark.parametrize("garbage", ["{trunc", "[]", '{"attempts": [{"bogus": 1}]}'])
def test_corrupt_state_ends_the_ladder(tmp_path, garbage):
    """Fix: an unreadable state used to restart the ladder from zero (resubmit forever)."""
    (tmp_path / "batch-jobs.log").write_text("", encoding="utf-8")
    run = tmp_path / "X_zn1"
    run.mkdir()
    path = rerun_state.state_path(run, "X_zn1")
    path.write_text(garbage, encoding="utf-8")
    state = rerun_state.load_state(path, "X_zn1")
    assert state.resolution == rerun_state.RESOLUTION_NEEDS_HUMAN and state.is_terminal
    assert (run / f"{path.name}.corrupt").read_text() == garbage
    # The verdict is persisted, so the next invocation does not start a fresh ladder,
    # and the batch log says so.
    assert rerun_state.load_state(path, "X_zn1").is_terminal
    assert "X_zn1\tNEEDS_HUMAN" in (tmp_path / "batch-jobs.log").read_text(encoding="utf-8")


def test_rerun_orca_does_not_resubmit_on_a_corrupt_state(tmp_path, fake):
    run_id = "COR.S2N2_zn1"
    run = _make_failed_run(tmp_path, run_id)
    (run / f"{run_id}-rerun-state.json").write_text("{trunc", encoding="utf-8")
    assert _rerun(run) == "terminal"
    assert fake.submits == []


# --------------------------------------------------------------------------- #
# xas-rerun-corvus and the CORVUS auto-rerun
# --------------------------------------------------------------------------- #
def _corvus_run(batch: Path, run_id: str, *, dead: bool = False) -> Path:
    """A split-layout run with a .hess and prior CORVUS output to archive."""
    working = batch / run_id / f"working-{run_id}"
    output = batch / run_id / f"output-{run_id}"
    feff = working / "Corvus3_cfavg_xas" / "Corvus1Zn_0_FEFF"
    for leg in ("xanes", "exafs"):
        (feff / leg).mkdir(parents=True)
    header = "#     0/   0 paths used\n#  omega e k mu mu0 chi\n"
    exafs_header = "#    14/  14 paths used\n#  omega e k mu mu0 chi\n"
    chi = "0.0" if dead else "1.0E-02"
    (feff / "xanes" / "xmu.dat").write_text(header + f" 9650 0 0 1 1 {chi}\n 9651 1 0.1 1 1 {chi}\n")
    (feff / "exafs" / "xmu.dat").write_text(exafs_header + " 9650 0 0 1 1 3e-2\n")
    (working / "Corvus.cfavg_xas.out").write_text(header + " 9650 0 0 1 1 1e-2\n 9651 1 0.1 1 1 2e-2\n")
    (working / f"{run_id}.hess").write_text("", encoding="utf-8")
    output.mkdir(parents=True)
    (output / f"xas-{run_id}.dat").write_text("old\n")
    return batch / run_id


@pytest.fixture
def corvus_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(rerun_corvus, "check_executable", lambda name: None)
    batch = tmp_path / "batch"
    for run_id in ("A_zn1", "B_zn1"):
        _corvus_run(batch, run_id)
    return batch


def test_rerun_corvus_records_per_run_submit_failures(corvus_batch, fake):
    """Fix: one refused submission aborted the loop / went unrecorded. It is now
    recorded per run, the others go ahead, and the postprocess waits only on
    the jobs that exist."""
    fake.fail_submit.add("generated-A_zn1-corvus-xas-wrapper.script")
    outcome = rerun_corvus.rerun_ids(corvus_batch, scheduler="slurm", tag="rerun-T")

    assert [r.run_id for r in outcome.records] == ["B_zn1"]
    assert set(outcome.failed) == {"A_zn1"}
    [b_job] = fake.by_script("generated-B_zn1-corvus-xas-wrapper.script")
    [post] = fake.by_script(f"generated-rerun-postprocess-{corvus_batch.name}-xas.script")
    assert post["afterok"] == [b_job["id"]] and post["afterany"] == []
    assert outcome.postprocess_job_id == post["id"]

    log_text = (corvus_batch / "batch-jobs.log").read_text()
    assert "rerun-corvus-xas-A_zn1\tSUBMIT_FAILED" in log_text
    assert f"rerun-corvus-xas-B_zn1\tSUBMITTED\tjob_id={b_job['id']}" in log_text
    state_text = outcome.state_file.read_text()
    assert "SUBMISSION FAILED: A_zn1" in state_text
    # Archived before the refusal: the next attempt starts clean all the same.
    working = corvus_batch / "A_zn1" / "working-A_zn1"
    assert (working / "Corvus3_cfavg_xas.rerun-T").is_dir()


def test_rerun_corvus_cli_exit_code_reflects_failures(corvus_batch, fake, monkeypatch):
    fake.fail_submit.add("generated-A_zn1-corvus-xas-wrapper.script")
    monkeypatch.setattr("sys.argv", ["xas-rerun-corvus", str(corvus_batch), "--scheduler", "slurm"])
    assert rerun_corvus.main() == 1


def test_rerun_corvus_postprocess_after_any(corvus_batch, fake):
    """Fix: postprocess_after lets an automated caller gate afterany."""
    outcome = rerun_corvus.rerun_ids(corvus_batch, scheduler="slurm", tag="rerun-T",
                                     postprocess_after="afterany")
    ids = [r.corvus_job_id for r in outcome.records]
    [post] = fake.by_script("-xas.script")
    assert post["afterany"] == ids and post["afterok"] == []


def test_rerun_corvus_refused_postprocess_is_recorded(corvus_batch, fake):
    fake.fail_submit.add(f"generated-rerun-postprocess-{corvus_batch.name}-xas.script")
    outcome = rerun_corvus.rerun_ids(corvus_batch, scheduler="slurm", tag="rerun-T")
    assert len(outcome.records) == 2 and outcome.postprocess_job_id is None
    assert f"rerun-postprocess-{corvus_batch.name}" in outcome.failed


def test_rerun_corvus_no_submit_touches_nothing(corvus_batch, fake):
    outcome = rerun_corvus.rerun_ids(corvus_batch, scheduler="slurm", tag="rerun-T", submit=False)
    assert {r.corvus_job_id for r in outcome.records} == {rerun_corvus.NO_SUBMIT}
    assert fake.calls == []
    assert (corvus_batch / "A_zn1" / "working-A_zn1" / "Corvus3_cfavg_xas").is_dir()
    assert not (corvus_batch / "batch-jobs.log").exists()


def test_auto_rerun_corvus_follow_up_is_afterany(tmp_path, fake, monkeypatch):
    """Fix: the follow-up postprocess was afterok, so a recompute that failed
    outright left it waiting forever and the run was never triaged again."""
    monkeypatch.setattr(rerun_corvus, "check_executable", lambda name: None)
    monkeypatch.delenv(auto_rerun_corvus.AUTO_RERUN_ENV, raising=False)
    batch = tmp_path / "batch"
    _corvus_run(batch, "DEAD_zn1", dead=True)
    _corvus_run(batch, "GOOD_zn1")
    (batch / layout.FAILED_CORVUS_MANIFEST).write_text("DEAD_zn1\n")
    monkeypatch.setattr("sys.argv", ["xas-auto-rerun-corvus", str(batch), "--scheduler", "slurm"])

    assert auto_rerun_corvus.main() == 0

    [job] = fake.by_script("-corvus-xas-wrapper.script")
    assert job["script"] == "generated-DEAD_zn1-corvus-xas-wrapper.script"
    [post] = fake.by_script(f"generated-rerun-postprocess-{batch.name}-xas.script")
    assert post["afterany"] == [job["id"]] and post["afterok"] == []
    state = rerun_state.load_state(rerun_state.state_path(batch / "DEAD_zn1", "DEAD_zn1", kind="corvus"),
                                   "DEAD_zn1")
    assert [a.corvus_job_id for a in state.attempts] == [job["id"]]
    assert layout.read_failed_corvus_ids(batch) == []
