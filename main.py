import argparse

parser = argparse.ArgumentParser(description='Semisupervised large image dataset filtering for crowdsourced classification')

parser.add_argument('--method', help='data filtering method, choose one: ALL, siamese, GAN, GANM, baseline',
                    default='ALL')
parser.add_argument('--case', help='dataset case, choose one: ALL, mj, case1, case2',
                    default='ALL')

def main():
    args = parser.parse_args()
    method = args.method
    case = args.case

    
    print(args)

if __name__ == '__main__':
    main()