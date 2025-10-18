class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset=='isic2018':
            return r'D:\KHTN2023\research25\HCT-Net\datasets'
        elif dataset=='cvc':
            return r'D:\KHTN2023\research25\HCT-Net\datasets'
        elif dataset=='chaos':
            return r'D:\KHTN2023\research25\HCT-Net\datasets'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
