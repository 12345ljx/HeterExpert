import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_xticks(np.arange(0.5, 8.5, 1))
ax.set_yticks(np.arange(0.5, 8.5, 1))
ax.set_xticklabels(['E{}'.format(i) for i in range(8)], ha='center')
ax.set_yticklabels(['D{}'.format(i) for i in range(7, -1, -1)])
ax.tick_params(axis='both', which='both', length=0, labelsize=18)

for x in range(9):
    ax.plot([x, x], [0, 8], color='black', lw=1)  
for y in range(9):
    ax.plot([0, 8], [y, y], color='black', lw=1)  

# ax.grid(which='both', color='black', linestyle='-', linewidth=2)
# check_positions = [(0,5), (1,7), (2,2), (3,1), (4,3), (5,4), (5,6), 
#                    (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (7,0)]

check_positions = [(0,2), (1,5), (1,6), (2,3), (3,0), (4,4), (5,7), (6,1),
                   (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7)]
for pos in check_positions:
    x, y = pos
    # if x in [0,1,4,5]:
    if x in [0,4,5]:
        color = 'red'
    else:
        color = 'black'
    ax.text(x + 0.5, y + 0.5, r'$\checkmark$', fontsize=27, ha='center', va='center', color=color)
    
plt.subplots_adjust(left=0.15, bottom=0.15)
ax.set_xlabel('Experts', fontsize=20)
ax.set_ylabel('Domains', fontsize=20)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f'/usr/workdir/HeterExpert/Figure/expert_choice/expert_choice2.png', bbox_inches='tight', 
            dpi=300)
plt.close()