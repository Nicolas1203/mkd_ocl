import argparse
import yaml

class Parser:
    """
    Command line parser based on argparse. This also includes arguments sanity check.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch UCL, with tensorboard logs.")

        # Configuration parameters
        parser.add_argument('--config', default=None, help="Path to the configuration file for the training to launch.")
        # Training parameters
        parser.add_argument('--train', dest='train', action='store_true')
        parser.add_argument('--test', dest='train', action="store_false")
        parser.add_argument('--epochs', default=1, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=10, type=int,
                            help='mini-batch size (default: 10)')
        parser.add_argument('--learning-rate', '-lr', default=0.0005, type=float, help='Initial learning rate')
        parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        parser.add_argument('--optim', default='Adam', choices=['Adam', 'SGD'])
        parser.add_argument('--head-reg-coef', type=float, default=0)
        parser.add_argument("--save-ckpt", action='store_true', help="whether to save chekpoints or not")
        parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
        parser.add_argument('--memory-only', '-mo', action='store_true', help='Training using only the memory ?')
        parser.add_argument('--alpha-sscl', type=float, help="Weight for stream in SSCL loss", default=1.0)
        # Logs parameters
        parser.add_argument('--tag', '-t', default='', help="Base name for graphs and checkpoints")
        parser.add_argument('--tb-root', default='./runs/', help="Where do you want tensorboards graphs ?")
        parser.add_argument('--logs-root', default='./logs', help="Defalt root folder for writing logs.")
        parser.add_argument('--results-root', default='./results/', help='Where you want to save the results ?')
        parser.add_argument('--tensorboard', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        # Early stopping
        parser.add_argument('--es-patience', help="early stoppping patience", type=int, default=10)
        parser.add_argument('--es-delta', type=float, help="minimum gap between each step for early stopping", default=0.01)
        # Boosting params
        parser.add_argument('--boost-lr', default=0.1, type=float, help='Learning rate for boosting')
        parser.add_argument('--scaling-coef', default=1, type=float, help='Scaling coef for boosting')
        parser.add_argument('--max-boosting-iter', type=int, default=1, help="Max projectors for MLBoost training.")
        parser.add_argument('--mlb-all', action='store_true', help="Use entire memory for mlboost")
        # checkpoints params
        parser.add_argument('--ckpt-root', default='./checkpoints/', help='Directory where to save the model.')
        parser.add_argument('--resume', '-r', action='store_true', 
                            help="Resume old training. Setup model state and buffer state.")
        parser.add_argument('--model-state')
        parser.add_argument('--buffer-state')
        ##########
        # MODELS #
        ##########
        parser.add_argument('--head', default='mlp')
        # Resnet parameters
        parser.add_argument('--proj-dim', type=int, default=128)
        parser.add_argument('--nb-channels', type=int, default=3, 
                            help="Number of channels for the input image.")
        parser.add_argument('--eval-proj', action='store_true', help="Use projection for inference. (default is representation.)")
        parser.add_argument('--pretrained', action='store_true', help="Use a pretrained model if available.")
        # Intermediate classifier
        parser.add_argument('--supervised', action='store_true', help="Pseudo labels or true labels ?")
        parser.add_argument('--dim-int', type=int, default=512)
        parser.add_argument('-nf', type=int, default=64, help="Number of feature for Resnet18. Set nf=20 for reduced resnet18, nf=64 for full.")
        #####################
        # Dataset parameters
        #####################
        parser.add_argument('--data-root-dir', default='/data/',
                            help='Root dir containing the dataset to train on.')
        parser.add_argument('--min-crop', type=float, default=0.2, help="Minimum size for cropping in data augmentation. range (0-1)")
        parser.add_argument('--dataset', '-d', default="cifar10",
                            choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'tiny', 'imagenet100', 'yt'],
                            help='Dataset to train on')
        parser.add_argument('--training-type', default='inc', choices=['uni', 'inc', 'blurry'],
                            help='How to feed the data to the network (incremental context or not)')
        parser.add_argument('--n-classes', type=int, default=10,
                            help="Number of classes in database.")
        parser.add_argument("--img-size", type=int, default=32, help="Size of the square input image")
        parser.add_argument('--num-workers', '-w', type=int, default=0, help='Number of workers to use for dataloader.')
        parser.add_argument("--n-tasks", type=int, default=5, help="How many tasks do you want ?")
        parser.add_argument("--labels-order", type=int, nargs='+', help="In which order to you want to see the labels ? Random if not specified.")
        parser.add_argument("--blurry-scale", type=int, default=3000)
        # Contrastive loss parameters
        parser.add_argument('--temperature', default=0.07, type=float, 
                            metavar='T', help='temperature parameter for softmax')
        # Semi supervised contrastive loss
        parser.add_argument('--kmeans-pl', action='store_true', help='Use kmeans as pseudo labels for SupCon loss.')
        parser.add_argument('--spectral', action="store_true", help="Use spectral clustering on representations as pseudo labels.")
        # A lot of gram parameters
        parser.add_argument('--gram-th', type=float, default=0, help='Threshold in gram matrix for SSCL loss')
        parser.add_argument('--gram-version', type=int, default=1)
        parser.add_argument('--p-before-gram', type=float, default=0, help="Wait to have seen some data before using gram matrix.")
        parser.add_argument('--gram-crit', type=float, default=1, help='Maximum value of gram matrix avg before using gram as pseudo label')
        parser.add_argument('--gram-mo', action='store_true', help="Computing gram matrix on memory only.")
        # Memory parameters
        parser.add_argument('--mem-size', type=int, default=200, help='Memory size for continual learning')  # used also for ltm
        parser.add_argument('--mem-batch-size', '-mbs', type=int, default=200, help="How many images do you want to retrieve from the memory/ltm")  # used also for ltm
        parser.add_argument('--buffer', default='reservoir', help="What buffer do you want ? See available buffers in utils/name_match.py")
        parser.add_argument('--drop-method', default='random', choices=['random'], help="How to drop images from memory when adding new ones.")
        # Pseudo labels parameters
        parser.add_argument('--classifier', choices=['ncm', 'knn'], default='ncm', help='Classifier to use for pseudo labelli ng')
        parser.add_argument('--random-pl', type=float, default=0,
                            help="Using random classifier to see how much you need it. range (0-1). If >0, other labels are ground truth")
        parser.add_argument('--random-pl-version', type=int, default=1,
                            help="0 for random labels from all classes. 1 for random labels from seen classes only.")
        parser.add_argument('--labels-ratio', type=float, default=0.03, help="Ratio of labeled data for SSL")
        # Distillation parameters
        parser.add_argument('--ocm-custom', action="store_true")
        # Learner parameter
        parser.add_argument('--learner', help='What learner do you want ? See list of available learners in utils/name_match.py')
        parser.add_argument('--debug', action='store_true')
        # Inference parameters
        parser.add_argument('--eval-mem', action='store_true', dest='eval_mem')
        parser.add_argument('--eval-random', action='store_false', dest='eval_mem')
        parser.add_argument('--lab-pc', type=int, default=20, help="Number of labeled images per class to use in unsupervised evaluation.")
        # Multi runs arguments
        parser.add_argument('--n-runs', type=int, default=1, help="Number of runs, with different seeds each time.")
        parser.add_argument('--start-seed', type=int, default=0, help="First seed to use.")
        parser.add_argument('--run-id', type=int, help="Id of the current run in multi run.")
        parser.add_argument('--kornia', action='store_true', dest='kornia')
        parser.add_argument('--no-kornia', action='store_false', dest='kornia')
        # Tricks
        parser.add_argument('--mem-iters', type=int, default=1, help="Number of times to make a grad update on memory at each step")
        parser.add_argument('--review', '-rv', action='store_true', help="Review trick on last memory.")
        parser.add_argument('--rv-iters', '-rvi', type=int, default=20, help="Number of iteration to do on last memory after training.")
        parser.add_argument('--tf-type', default="full", choices=['full', 'partial', 'gsa'], help=' Data augmentation sequence to use.')
        # Multi aug params || DAA params
        parser.add_argument('--n-augs', type=int, default=1)
        parser.add_argument('--n-styles', type=int, default=0)
        parser.add_argument("--tf-size", type=int, default=128)
        parser.add_argument("--min-mix", type=float, default=0.5, help="Min proportion of the original image to keep when using mixup or cutmix.")
        parser.add_argument("--max-mix", type=float, default=1.0, help="Max proportion of the original image to keep when using mixup or cutmix.")
        parser.add_argument('--mixup', action='store_true')
        parser.add_argument('--n-mixup', type=int, default=0, help='Numbers of mixup to consider.')
        parser.add_argument('--cutmix', action='store_true')
        parser.add_argument('--n-cutmix', type=int, default=0, help='Numbers of cutmix to consider.')
        parser.add_argument('--jfmix', action='store_true')
        parser.add_argument('--jfmix-samples', type=int, default=5)
        parser.add_argument('--saliency', action='store_true')
        parser.add_argument('--fmix', action='store_true')
        parser.add_argument('--save-cluster-dist', action='store_true')
        parser.add_argument('--aug-version', type=int, default=0)
        parser.add_argument('--p-strong', type=float, default=0.8, help="Probability for using a strong aug.")
        parser.add_argument('--multi-style', action='store_true')
        parser.add_argument('--ycbcr-ratio', type=float, default=1, help="between 0 and 1")
        parser.add_argument('--daa', dest='daa', action='store_true', help="Using DAA")
        parser.add_argument('--va', dest='daa', action="store_false", help="Using Vanilla augmentations")
        parser.add_argument("--style-samples", type=int, default=1)
        parser.add_argument("--min-style-alpha", type=float, default=1)
        parser.add_argument("--max-style-alpha", type=float, default=1)
        # GML arguments
        parser.add_argument('--var', type=float, default=0.14, help="Variance for gml loss")
        parser.add_argument('--norm-mean', action='store_true', help='GML project means to unit sphere when training.')
        parser.add_argument("--gml-loss", help='Loss to optimize with gml.', default='map')
        parser.add_argument('--gml-implem', help="old or new implementation.", default='new')
        parser.add_argument("--fixed-means", dest='fixed_means', action='store_true', help='Using fixed means for GML.')
        parser.add_argument("-C", type=float, help='Hyperparams if using both loss for gml', default=0)
        parser.add_argument("--mean-pc", action='store_true', help="Using means per class for gml loss.")
        parser.add_argument('--eval-mean', action='store_true', help='Using fixed means for evaluation.')
        parser.add_argument('--relaxed', action='store_true')
        parser.add_argument('--tot-classes-gml', type=int, default=0)
        parser.add_argument('--all-means', action='store_true')
        parser.add_argument('--plot-gml', action='store_true')
        parser.add_argument('--gen-strat', help="Artificial data generation strategy for GML. Choices=[neg, pos, all]", default=None)
        parser.add_argument('-mu', type=float, default=1.0, help="mu value for GML model.")
        # DERpp arguments
        parser.add_argument('--derpp-alpha', type=float, default=0.1, help="Values of alpha un der++ loss")
        parser.add_argument('--derpp-beta', type=float, default=0.5, help="Values of beta un der++ loss")
        # MC-DKD arguments
        parser.add_argument('--alpha-ce', type=float, default=1)
        parser.add_argument('--alpha-kd', type=float, default=1)
        parser.add_argument('--beta-ce', type=float, default=1)
        parser.add_argument('--beta-kd', type=float, default=1)
        parser.add_argument('--gamma-ce', type=float, default=1)
        parser.add_argument('--gamma-kd', type=float, default=1)
        parser.add_argument('--kd-lambda', type=float, default=8)
        parser.add_argument('--kd-temperature', type=float, default=4)
        parser.add_argument('--kd-kappa', type=float, default=1)
        parser.add_argument('--kd-strat', type=int, default=0)
        parser.add_argument('--kd-scheduler', action='store_true', help="schedule the strenght of the distillation over time.")
        parser.add_argument('--vanilla-beta', action='store_true', help='Use regular probabilities as beta')
        parser.add_argument('--ema', action='store_true')
        parser.add_argument('--group-selection', default="all", help="How to selct differents groups of labels for sep CE and KL.")
        parser.add_argument('--inter-focal-ce', action='store_true')
        parser.add_argument('--intra-focal-ce', action='store_true')
        parser.add_argument('--inter-focal-kd', action='store_true')
        parser.add_argument('--intra-focal-kd', action='store_true')
        parser.add_argument('--kd-new', action='store_true')
        parser.add_argument('--beta-min-value', default="adaptative", help="adaptative of constant")
        parser.add_argument('--eval-teacher', action='store_true')
        parser.add_argument('--drop-fc', action="store_true")
        parser.add_argument('--last-t-kd', action='store_true')
        parser.add_argument('--dist-strat', default='vanilla')
        parser.add_argument('--binary-dist', action='store_true')
        parser.add_argument('--sd-strat', default="vanilla")
        # EMA params
        parser.add_argument('--ema-alpha', type=float, default=0.1)
        parser.add_argument('--ema-beta', type=float, default=0.01)
        parser.add_argument('--ema-beta2', type=float, default=0.005)
        parser.add_argument('--ema-correction-step', type=int, default=10)
        parser.add_argument('--wema', default="logits")
        parser.add_argument('--maxmix', type=float, default=1)
        parser.add_argument('--tm-strat', type=str, default="1tm")
        parser.add_argument('--n-assistants', type=int, default=1)
        parser.add_argument('--mkd-strat', default='auto_avg')
        parser.add_argument('--mkd-loss', default='vanilla')
        parser.add_argument('--agg-model-search', type=int, default=-1)
        # MMKD params
        parser.add_argument('--ema-alpha1', type=float, default=0.1)
        parser.add_argument('--ema-alpha2', type=float, default=0.05)   
        parser.add_argument('--ema-alpha3', type=float, default=0.01)
        parser.add_argument('--ema-alpha4', type=float, default=0.005)
        parser.add_argument('--ema-alpha5', type=float, default=0.0001)
        parser.add_argument('--avg-ema', action='store_true', help="Takes the average of EMA models for inference.")
        # MMKD with a better alpha search
        parser.add_argument('--alpha-min', type=float, default=0.01)
        parser.add_argument('--alpha-max', type=float, default=0.01)
        parser.add_argument('--n-teacher', type=int, default=1)
        # SDP params
        parser.add_argument('--sdp-mu', type=int, default=100)
        parser.add_argument('--sdp-c2', type=float, default=0.75)
        
        # Cosine annealing
        parser.add_argument('--annealing', action='store_true')
        parser.add_argument('--annealing-reset-interval', type=int, default=100)
        # Zeta Mixup
        parser.add_argument('--zeta-gamma', type=float, default=2.4)
        # adaptative mixup
        parser.add_argument('--ada-mixup', type=int, default=1)
        # Gdumb arguments
        parser.add_argument('--custom-gdumb', action='store_true')
        # Instance normalization
        parser.add_argument('--instance-norm', action='store_true')
        # WandB argument
        parser.add_argument("--no-wandb", action='store_true')
        parser.add_argument('--wandb-watch', action='store_true', help="Watch the models gradient and parameters into wandb (can be slow).")
        parser.add_argument("--sweep", action='store_true', help="Run the code with sweep for HP search.")
        parser.set_defaults(train=True, eval_mem=True, kornia=True, daa=True, fixed_means=True)
        # temporary arguments
        parser.add_argument('--measure-drift', type=int, default=-1)
        parser.add_argument('--tsne', action='store_true')
        parser.add_argument('--no-aug', action="store_true")
        parser.add_argument('--no-avg', action="store_true")
        self.parser = parser

    def parse(self, arguments=None):
        if arguments is not None:
            self.args = self.parser.parse_args(arguments)
        else:
            self.args = self.parser.parse_args()
        self.load_config()
        self.check_args()
        return self.args

    def load_config(self):
        if self.args.config is not None:
            with open(self.args.config, 'r') as f:
                cfg = yaml.safe_load(f)
                for key in cfg:
                    setattr(self.args, key, cfg[key])
            f.close()

    def check_args(self):
        """Modify default arguments values depending on the method and dataset.
        """
        #############################
        # Dataset parameters sanity #
        #############################
        if self.args.dataset == 'cifar10':
            self.args.img_size = 32
            self.args.n_classes = 10
            self.args.dim_in = 8 * self.args.nf
        if self.args.dataset == 'cifar100': 
            self.args.img_size = 32
            self.args.n_classes = 100
            self.args.dim_in = 8 * self.args.nf
        if self.args.dataset == 'tiny':
            self.args.img_size = 64
            self.args.n_classes = 200
            self.args.dim_in = 32 * self.args.nf            
            # Style parameters specific to tinyIN (TODO: add this to configs)
            self.min_style_alpha=0.4
            self.max_style_alpha=0.8
        if self.args.dataset == 'yt':
            self.args.img_size = 224
            self.args.n_classes = 2
            self.args.dim_in = 392 * self.args.nf
        if self.args.dataset == 'imagenet100':
            self.args.img_size = 224
            self.args.n_classes = 100
            self.args.dim_in = 8 * self.args.nf

        ##############################
        # Learners parameters sanity #
        ##############################
        if self.args.learner == 'STAM':
            self.args.batch_size = 1

