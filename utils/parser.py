import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="MCCLK")
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="music", help="Choose a dataset:[last-fm,amazon-book, alibaba, book, music, movie]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train ===== # test
    parser.add_argument('--epoch', type=int, default=400, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')  # default = 1e-4, lr=1e-3 is bad for music, 3e-3 is better.
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    # parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument('--inverse_r', dest='inverse_r', action='store_true')
    parser.add_argument('--no_inverse_r', dest='inverse_r', action='store_false')
    parser.set_defaults(inverse_r=True)
    # parser.add_argument("--node_dropout", type=bool, default=0, help="consider node dropout or not")
    parser.add_argument('--node_dropout', dest='node_dropout', action='store_true')
    parser.add_argument('--no_node_dropout', dest='node_dropout', action='store_false')
    parser.set_defaults(node_dropout=True) # 防止模型过拟合, 非常有效:
    # parser.set_defaults(node_dropout=False) # test_auc:0.8700, test_f1:0.7757
    # parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout") # original version
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout") # best
    # parser.add_argument("--mess_dropout", type=bool, default=0, help="consider message dropout or not")
    parser.add_argument('--mess_dropout', dest='mess_dropout', action='store_true')
    parser.add_argument('--no_mess-dropout', dest='mess_dropout', action='store_false')
    parser.set_defaults(mess_dropout=True) # 防止模型过拟合, 非常有效;
    # parser.set_defaults(mess_dropout=False) # test_auc:0.8700, test_f1:0.7757
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--ind", type=str, default='mi', help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
    parser.add_argument("--cl_w", type=float, default=0.1, help="the weight for contrastive learning")

    return parser.parse_args()
