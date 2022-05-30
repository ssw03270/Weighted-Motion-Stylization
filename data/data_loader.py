f = open('contents.txt', 'r')
contents = [line.strip() for line in f.readlines()]

f = open('styles.txt', 'r')
styles = [line.strip() for line in f.readlines()]

def create_data_loader(type='train'):
    print('create source dataset %s phase...' % type)

    if type == 'train':
