import numpy as np
import time

padding = 0
scale = 1
from PIL import Image, ImageDraw

MIN_BOX = 10 ** 9


def color_avg(hist):
    total = sum(hist)
    value, error = 0, 0
    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5
    return value, error


def hist_to_clor(hist):
    l, e = color_avg(hist)
    return int(l), e


class QtreeNode(object):

    def __init__(self, img, box, depth):
        global MIN_BOX
        self.box = box
        self.depth = depth
        self.children = None
        self.leaf = False
        im = img.crop(box)
        self.width, self.height = im.size
        hist = im.histogram()
        self.color, self.error = hist_to_clor(hist)

    def IsLeaf(self):
        return self.leaf

    def NodeSplit(self, img):
        l, t, r, b = self.box
        lr = l + (r - l) / 2
        tb = t + (b - t) / 2
        top_l = QtreeNode(img, (l, t, lr, tb), self.depth + 1)
        top_r = QtreeNode(img, (lr, t, r, tb), self.depth + 1)
        bottom_l = QtreeNode(img, (l, tb, lr, b), self.depth + 1)
        bottom_r = QtreeNode(img, (lr, tb, r, b), self.depth + 1)
        self.children = [top_l, top_r, bottom_l, bottom_r]


class Qtree(object):

    def __init__(self, im, quality, max_depth=1024):
        self.root = QtreeNode(im, im.getbbox(), 0)
        self.width, self.height = im.size
        self.max_depth = 0
        self.data = [[], [], [], []]  # x ; y ; size ; color
        self.threshold = 3
        self._TreeBuild_(im, self.root, max_depth)

    def _TreeBuild_(self, im, node, max_depth):

        if (node.depth >= max_depth) or (node.error <= self.threshold):
            if node.depth > self.max_depth:
                self.max_depth = node.depth
            node.leaf = True
            self.data[0].append(int(node.box[0]))
            self.data[1].append(int(node.box[1]))
            self.data[2].append(int(abs(node.box[0] - node.box[2])))
            self.data[3].append(int(node.color))
            return

        node.NodeSplit(im)
        for child in node.children:
            self._TreeBuild_(im, child, max_depth)

    def GetLeafs(self, depth):

        def GetLeafsRecurs(tree, node, depth, func):
            if node.leaf is True or node.depth == depth:
                func(node)
            elif node.children is not None:
                for child in node.children:
                    GetLeafsRecurs(tree, child, depth, func)

        if depth > tree.max_depth:
            raise ValueError('A depth larger than the trees depth was given')

        leaf_nodes = []
        GetLeafsRecurs(self, self.root, depth, leaf_nodes.append)
        return leaf_nodes

    def ImgDepth(self, depth):

        m = scale
        dx, dy = (padding, padding)
        im = Image.new('L', (int(self.width * m + dx), int(self.height * m + dy)))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, self.width * m + dx, self.height * m + dy), 0)

        leaf_nodes = self.GetLeafs(depth)
        for node in leaf_nodes:
            l, t, r, b = node.box
            box = (l * m + dx, t * m + dy, r * m - 1, b * m - 1)
            draw.rectangle(box, node.color)
        return im

    def Render(self, depth=0):
        if depth > self.max_depth:
            raise ValueError('A depth larger than the trees depth was given')

        im = self.ImgDepth(depth)
        im.save("outFractal.bmp")


class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def str(self):
        return '%s_%s' % (self.left, self.right)


def HuffmanTree(node, left=True, binString=''):
    if type(node) is int:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(HuffmanTree(l, True, binString + '0'))
    d.update(HuffmanTree(r, False, binString + '1'))
    return d


start = time.time()
im = Image.open('lena.bmp').convert('L')
im.save("lena_grayscale.bmp")
tree = Qtree(im, quality=100)
symbols = np.unique(tree.data[3])

freq = {}
for i in tree.data[3]:
    if i in freq:
        freq[i] += 1
    else:
        freq[i] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
nodes = freq

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))

    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = HuffmanTree(nodes[0][0])

huffman_weight = 0
for x in tree.data[3]:
    huffman_weight += len(huffmanCode[x])
end = time.time()
orig = im.size[0] * im.size[1] * 8
res = im.size[0] * im.size[1] + huffman_weight
print(orig / res)
print((end - start))

print(tree.max_depth)
tree.Render(tree.max_depth)
