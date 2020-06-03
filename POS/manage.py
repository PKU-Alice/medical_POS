import argparse


def preprocess():
    from DataProcessing.preprocess import preprocess
    preprocess()

def pos_train():
    pass

def pos_eval():
    pass

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('--preprocess',action='store_true',help='数据预处理')
    group.add_argument('--pos_train',action='store_true',help='词性标注模型训练')
    group.add_argument('--pos_eval',action='store_true',help='词性标注模型评价')

    args = parser.parse_args()

    if args.preprocess:
        preprocess()
    elif args.pos_train:
        pos_train()
    elif args.pos_eval:
        pos_eval()


if __name__ == '__main__':
    main()