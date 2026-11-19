# mopd_3593882 hopd-p55k — sacct FAILED 04:19:17 exit 1:0 BUT metrics through step 500 + val
# 55k data × 500 steps × resp 128 × bs2 (capped; not full epoch)
# Final val_acc=0.531 (n=704): MathVision 0.688 · MathVerse 0.500 · MathVista 0.325
# first→last: loss 5.46→4.08 | entropy 0.40→8.52 | student_mass 0.008→0.081
# Check .out for post-train failure (teardown / Adam after val / Ray kill).
