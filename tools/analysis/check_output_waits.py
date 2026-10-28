"""Run existing ownership checks plus storer observer partition checks."""
import tools.analysis.check_waits as base
checked=0
for rows,ops in zip(base.diag,base.ops):
 for r,op in zip(rows,ops):
  if op not in (1,4,5,6,7):continue
  assert len(r)==18
  if not base.args.allow_clock_range_overflow:
   assert sum(r[12:15])<=r[15] and r[16]+r[17]<=r[13],r
  checked+=1
assert checked==3028
print('Output observer ownership and applicable clock checks PASS:',checked)
