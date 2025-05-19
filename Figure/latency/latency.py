import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

flops = [815665536, 1593245728, 2456954240, 3232568032, 4024896224, 4819584096, 5639441888, 6447697920]
flops = [i / 1e9 for i in flops]  # Convert to billions
naive_time = [0.097472, 0.140096, 0.1952, 0.234688, 0.27573, 0.291776, 0.345984, 0.342784]
triton_time = [0.050214, 0.086592, 0.12384, 0.160512, 0.200233, 0.23936, 0.263262, 0.288129]
mlp_time = 0.273534

plt.figure(figsize=(8, 6))
plt.grid(True, linestyle='--', alpha=0.6, color='lightgray')

(line1,) = plt.plot(flops[-1], mlp_time, color='black', linestyle='', marker='x', markersize=8, markeredgewidth=3)
(line2,) = plt.plot(flops, naive_time, color='#ff7f0e', linestyle='--', marker='s', markersize=8, linewidth=2)
(line3,) = plt.plot(flops, triton_time, color='#2ca02c', linestyle='-.', marker='^', markersize=8, linewidth=2)

plt.xlabel('FLOPs (1e9)', fontsize=18)
plt.ylabel('Wall-clock time (ms)', fontsize=18)
plt.xticks(np.arange(0, 7.2, 1), fontsize=15)
plt.yticks(np.arange(0, 0.36, 0.05), fontsize=15)
plt.legend([line1, line2, line3], 
           ['MLP', 'HEMoE (naive)', 'HEMoE (Triton)'],
           loc='lower right', fontsize=15)


aux_line_width = 1.5

plt.axvline(x=flops[3], color='black', linestyle='--', linewidth=aux_line_width)
plt.plot([flops[3] - 0.3, flops[-1]], [mlp_time, mlp_time], color='blue', linestyle='--', linewidth=aux_line_width)
plt.plot([flops[3] - 0.3, flops[3]], [triton_time[3], triton_time[3]], color='blue', linestyle='--', linewidth=aux_line_width)

plt.gca().add_patch(FancyArrowPatch(posA=(flops[3], 0.348), 
                                    posB=(flops[3] + 0.28, 0.348),
                                    color='black', linestyle='-', arrowstyle='<-',
                                    mutation_scale=15, linewidth=aux_line_width))

plt.gca().add_patch(FancyArrowPatch(posA=(flops[3] - 0.2, triton_time[3]), 
                                    posB=(flops[3] - 0.2, mlp_time),
                                    color='blue', linestyle='-', arrowstyle='<->',
                                    mutation_scale=15, linewidth=aux_line_width))

plt.text(0.38, 0.56, '41% speedup', ha='right', va='top', transform=plt.gca().transAxes,
         fontsize=15, linespacing=1.5, color='blue')

plt.text(0.47, 0.96, 'Expert usage of HEMoE with\n96.5% original accuracy',
         ha='left', va='top', transform=plt.gca().transAxes,
         fontsize=15, linespacing=1.5)

# plt.gca().text(7.2, -0.07, '1e9', 
#               transform=plt.gca().get_xaxis_transform(),
#               ha='right', va='top', 
#               fontsize=15, color='black')

plt.xlim(0.6, 6.8)
plt.ylim(0.03, 0.38)

plt.savefig(f"/usr/workdir/HeterExpert/Figure/latency/latency.pdf", format='pdf')
plt.close()