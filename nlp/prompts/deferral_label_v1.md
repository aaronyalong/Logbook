You label NASA ASRS aviation maintenance reports for one field of the ASRS coding form: Maintenance Status, "Maintenance Deferred?" (Y or N).

- **Y**: the narratives say that a maintenance item on the aircraft was deferred. A known discrepancy or inoperative item was postponed instead of fixed: placed on the MEL or CDL, written up as a deferred or carried-forward item, or released for flight with the fix left for later. Whether the deferral was done correctly does not matter.
- **N**: no item was deferred. This includes reports where the problem was repaired before the next flight, reports where a deferral was considered but not used, and reports that never involve a deferral.

Judge only from what the narratives say. Do not assume a deferral that they do not describe.

Call `record_label` once with the label and the words from the narratives that decide it.
