# G1 PC2 latency recording

This launcher targets the existing PC2 runtime at `/home/unitree/MotrixLab` and its
`g1-wbt/hardware` recipe. It uses an isolated instrumented native build under
`/home/unitree/g1-sbto005-latency`, without replacing the installed transport.

On PC2:

```bash
bash /home/unitree/MotrixLab/run_sbto005_mass_com_pc2.sh eth0
```

Start ramps to motion frame zero in two seconds. A enables policy takeover and
playback. The 495-frame clip runs once at 50 Hz, then enters damping. The launcher
explicitly disables final-frame holding. Existing state safety gates remain active.

Files are saved under `/home/unitree/MotrixLab/logs/wbt/`:

- `.log`: existing phase and error messages.
- `.timing.npz`: raw monotonic timestamps, joint commands and feedback.
- `.timing.json`: units, column descriptions, sample counts and percentile summaries.

The native callback arrival timestamp is associated with the state consumed by
the task. Each command update is associated with the first successful native
DDS publish call. Its return timestamp means the DDS API accepted the write,
not robot receipt or motor execution. Repeated 1 kHz sends are recorded separately
from 50 Hz policy updates. Both writer and policy period statistics are included.

Observation age here means PC2 callback arrival to task entry; it excludes the
unknown sensor-to-PC2 delay. ONNX timing includes the existing policy inference
method. Command and 1 kHz feedback traces support later response-lag estimation;
mechanics, PD control and feedback transport affect that estimate.

Control-time recording uses bounded memory, with no per-sample printing or file
writes. Feedback and writer ring buffers retain the last 100,000 records (about
100 seconds at 1 kHz); slower streams retain their first 100,000 records.
Overwritten or dropped records are counted. Files
are written after CLI cleanup; SIGKILL or power loss can lose the buffered data.
Recording adds timestamp/copy/locking overhead; no hardware A/B overhead benchmark
has been performed.

`native-timing.patch` records the instrumentation against the PC2's installed
native source, which differs from the current local workspace. It belongs to
this PC2 recipe rather than a general workspace runtime migration.
