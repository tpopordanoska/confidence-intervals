import warnings
from cieg.experiments import *
from cieg.utils import create_folders

warnings.filterwarnings(
    "ignore", category=UserWarning
)

census_columns = ["iClass", "dIncome1", "iEnglish", "dHours"]
higgs_first = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv']
higgs_second = ['m_bb', 'm_wbb', 'm_wwbb']
osteoarthritis = ['Side', 'WOMAC', 'OSTM']  # OSTM OSFM


EXPERIMENTS = {
    'higgs_first': Higgs(usecols=higgs_first, name='Higgs first'),
    'higgs_second': Higgs(usecols=higgs_second, name='Higgs second'),
    'osteoarthritis': Osteoarthritis(usecols=osteoarthritis)
}


def run():
    for exp_name in EXPERIMENTS.keys():
        print(f"Running {exp_name}")
        path = create_folders()
        experiment = EXPERIMENTS[exp_name]
        experiment.run(path)


if __name__ == '__main__':
    run()
