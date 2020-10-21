import warnings
from cieg.experiments import *
from cieg.utils import create_folders

warnings.filterwarnings(
    "ignore", category=UserWarning
)

oa = ['WOMAC', 'OSTM', 'OSFM']
higgs_first = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv']
higgs_second = ['m_bb', 'm_wbb', 'm_wwbb']
higgs_third = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']


EXPERIMENTS = {
    'oa': Osteoarthritis(usecols=oa),
    'higgs_first': Higgs(usecols=higgs_first, name='higgs_first'),
    'higgs_second': Higgs(usecols=higgs_second, name='higgs_second'),
    'higgs_third': Higgs(usecols=higgs_third, name='higgs_third')
}


def run():
    for exp_name in EXPERIMENTS.keys():
        print(f"Running {exp_name}")
        path = create_folders()
        experiment = EXPERIMENTS[exp_name]
        experiment.run(path)


if __name__ == '__main__':
    run()
