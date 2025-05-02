from collections import namedtuple
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']  # 'Arial' 'Helvetica'

data = {
    'arc_easy': {
        'ilp_module': {
            "sparsity": [0.57603, 0.62160, 0.66563, 0.71769, 0.75812, 0.80553, 0.85756],
            "accuracy": [0.41793, 0.43140, 0.45202, 0.49158, 0.50631, 0.53241, 0.56818],
        },
        'ilp_module_stable': {
            "sparsity": [0.57160, 0.61132, 0.65988, 0.70327, 0.76904, 0.79489, 0.89076],
            "accuracy": [0.40783, 0.42593, 0.45539, 0.48148, 0.50631, 0.52020, 0.58544],
        },
        'dynk_max': {
            "sparsity": [0.58971, 0.61965, 0.65398, 0.69321, 0.85757],
            "accuracy": [0.41667, 0.41835, 0.42677, 0.45707, 0.52441],
        },
        'cluster': {
            "sparsity": [0.57236, 0.63345, 0.69454, 0.75563, 0.81673, 0.87782, 0.93891],
            "accuracy": [0.40152, 0.41204, 0.45749, 0.50547, 0.54714, 0.56902, 0.61742],
        },
        'original': 0.60648
    },
    'arc_challenge': {
        'ilp_module': {
            "sparsity": [0.56493, 0.61659, 0.66010, 0.69807, 0.75233, 0.80015, 0.84654],
            "accuracy": [0.25427, 0.27730, 0.27645, 0.29266, 0.31314, 0.31911, 0.343],
        },
        'ilp_module_stable': {
            "sparsity": [0.56816, 0.61129, 0.64989, 0.68815, 0.74299, 0.78900, 0.88964],
            "accuracy": [0.25000, 0.27048, 0.27645, 0.28413, 0.29437, 0.31570, 0.36177],
        },
        'dynk_max': {
            "sparsity": [0.61721, 0.64377, 0.69795, 0.78548, 0.85137],
            "accuracy": [0.26365, 0.27133, 0.27816, 0.28328, 0.33191],
        },
        'cluster': {
            "sparsity": [0.57236, 0.63345, 0.69454, 0.75563, 0.81673, 0.87782, 0.93891],
            "accuracy": [0.25085, 0.26792, 0.27218, 0.29096, 0.30631, 0.34386, 0.35666],
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
plt.plot(data[task_name]['ilp_module_stable']['sparsity'], data[task_name]['ilp_module_stable']['accuracy'], label='HEMoE', marker='x')
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