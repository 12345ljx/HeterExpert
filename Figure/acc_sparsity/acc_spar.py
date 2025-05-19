from collections import namedtuple
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

data = {
    'arc_easy': {
        'ilp': {
            "sparsity": [0.56638, 0.60859, 0.66142, 0.70978, 0.75489, 0.80095, 0.88844],
            "accuracy": [0.43140, 0.45623, 0.50589, 0.53662, 0.56776, 0.59049, 0.64815],
        },
        'cluster': {
            "sparsity": [0.57236, 0.63345, 0.69454, 0.75563, 0.81673, 0.87782, 0.93891],
            "accuracy": [0.42971, 0.47727, 0.51094, 0.56061, 0.59975, 0.64646, 0.69276],
        },
        'original': 0.60648
    },
    'arc_challenge': {
        'ilp': {
            "sparsity": [0.55806, 0.61621, 0.65204, 0.70309, 0.75407, 0.79248, 0.88767],
            "accuracy": [0.28328, 0.29352, 0.30973, 0.32850, 0.33447, 0.35239, 0.37799],
        },
        'cluster': {
            "sparsity": [0.57236, 0.63345, 0.69454, 0.75563, 0.81673, 0.87782, 0.93891],
            "accuracy": [0.27986, 0.28413, 0.30887, 0.33020, 0.34898, 0.36860, 0.40273],
        },
        'original': 0.36348
    },
    'hellaswag': {
        'ilp_module': {
            "sparsity": [0.55640, 0.60241, 0.64812, 0.69308, 0.73714, 0.78617, 0.83381],
            "accuracy": [0.374, 0.38857, 0.39343, 0.40914, 0.42343, 0.43429, 0.46714],
        },
        'ilp_module_stable': {
            "sparsity": [0.55434, 0.59433, 0.63788, 0.67792, 0.72656, 0.78217, 0.88313],
            "accuracy": [0.37686, 0.38029, 0.39743, 0.40629, 0.42743, 0.45114, 0.50229],
        },
        'cluster': {
            "sparsity": [0.57236, 0.63345, 0.69454, 0.75563, 0.81673, 0.87782, 0.93891],
            "accuracy": [0.37743, 0.40029, 0.418, 0.44343, 0.46771, 0.50143, 0.53714],
        },
        'original': 0.56714
    },
    'piqa': {
        'ilp_module': {
            "sparsity": [0.56255, 0.61494, 0.65915, 0.70304, 0.74812, 0.79267, 0.84131],
            "accuracy": [0.61752, 0.63166, 0.65071, 0.65887, 0.66268, 0.68444, 0.71436],
        },
        'ilp_module_stable': {
            "sparsity": [0.55827, 0.60430, 0.64639, 0.69012, 0.72901, 0.78803, 0.88350],
            "accuracy": [0.61534, 0.63656, 0.63874, 0.64527, 0.67084, 0.68988, 0.72144],
        },
        'dynk_max': {
            "sparsity": [0.63141, 0.65161, 0.67732, 0.69196, 0.74464, 0.75936],
            "accuracy": [0.62622, 0.62568, 0.62405, 0.63112, 0.63166, 0.63439],
        },
        'cluster': {
            "sparsity": [0.57236, 0.63345, 0.69454, 0.75563, 0.81673, 0.87782, 0.93891],
            "accuracy": [0.60881, 0.62405, 0.65397, 0.67193, 0.68607, 0.70892, 0.75462],
        },
        'original': 0.74483
    },
    'winogrande': {
        'ilp_module_stable': {
            "sparsity": [0.55693, 0.59848, 0.64552, 0.69077, 0.73233, 0.78441, 0.88610],
            "accuracy": [0.52802, 0.52092, 0.55012, 0.56827, 0.58327, 0.60616, 0.64325],
        },
        'dynk_max': {
            "sparsity": [0.62119, 0.65645, 0.68170, 0.70855, 0.74417],
            "accuracy": [0.51302, 0.52170, 0.55249, 0.57301, 0.57537],
        },
        'original': 0.60695
    },
}

task_name = 'arc_challenge'
label = 'ARC-c'
# task_name = 'piqa'
# label = 'PIQA'

plt.figure(figsize=(6, 5))
# plt.plot(data[task_name]['ilp_module']['sparsity'], data[task_name]['ilp_module']['accuracy'], label='ILP(module)', marker='x')
plt.plot(data[task_name]['ilp']['sparsity'], data[task_name]['ilp']['accuracy'], label='HEMoE', marker='x')
plt.plot(data[task_name]['cluster']['sparsity'], data[task_name]['cluster']['accuracy'], label='MoEfication', marker='x')

# plt.plot(data[task_name]['ilp_module_stable']['sparsity'], data[task_name]['ilp_module_stable']['accuracy'], label='top_k', marker='x')
# plt.plot(data[task_name]['dynk_max']['sparsity'], data[task_name]['dynk_max']['accuracy'], label='dynk_max', marker='x')
plt.axhline(y=data[task_name]['original'], color="#1085cd", linestyle="--", label='Original')

plt.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.90)

plt.tick_params(axis='both', labelsize=16)
plt.xlabel('Sparsity', fontsize=20)
plt.ylabel('Performance', fontsize=20)
# plt.ylim(0.5, 0.8)
# plt.xlim(0.5, 1.0)
plt.title(f'{label}', fontsize=20)
plt.legend(fontsize=16)
plt.savefig(f'/usr/workdir/HeterExpert/Figure/acc_sparsity/Spar_Acc_{task_name}.pdf', format='pdf')
plt.close()