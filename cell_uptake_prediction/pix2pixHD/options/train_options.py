from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results')
        self.parser.add_argument('--display_freq_epochs', type=int, default=10,
                                 help='save visualization images every N epochs')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving latest model')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load pretrained model from specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='do not use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='do not use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do not use least square GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='size of image buffer for fake images')

        # new (master-6): optimizer / scheduler
        self.parser.add_argument('--use_adamw', action='store_true', default=True,
                                 help='use AdamW optimizer (decoupled weight decay)')
        self.parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay for optimizer')
        self.parser.add_argument('--lr_scheduler', type=str, default='cosine',
                                 choices=['linear', 'cosine', 'none'], help='learning rate scheduler')
        self.parser.add_argument('--T_max', type=int, default=400, help='cosine scheduler T_max')
        self.parser.add_argument('--eta_min', type=float, default=1e-06, help='cosine scheduler minimum learning rate')
        self.parser.add_argument('--warmup_epochs', type=int, default=5, help='linear warmup epochs before main scheduler')

        # new (master-6): validation
        self.parser.add_argument('--val_freq_epochs', type=int, default=5, help='run validation every N epochs')
        self.parser.add_argument('--val_phase', type=str, default='val', help='dataset phase for validation')
        self.parser.add_argument('--val_batchSize', type=int, default=1, help='batch size for validation')

        self.isTrain = True
