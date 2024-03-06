

from collections import OrderedDict


class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def nodes(self):
        return self.left, self.right

    def __str__(self):
        return ' %s_%s \n' % (self.left, self.right)


def get_sorted_freq(arr):
    """create dictionary of frequency of characters and sort it"""

    # create frequency dictionary
    dict = OrderedDict.fromkeys(arr, 0)
    for n in arr: dict[n] += 1

    # sort dictionary according to frequency
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    return dict


def huffman_code_tree(node, bin_string=''):
    """takes in top node of the huffman code tree to create binary representations of characters"""

    if type(node) is str or type(node) is int:
        return {node: bin_string}

    (l, r) = node.children()

    d = dict()
    d.update(huffman_code_tree(l, bin_string + '0'))
    d.update(huffman_code_tree(r, bin_string + '1'))

    return d


def huffman_coding(arr):
    """
    create huffman coding tree and bytes to represent the compressed arr
    according to https://www.programiz.com/dsa/huffman-coding
    """

    nodes = get_sorted_freq(arr)
    # print(arr)
    # print(nodes)

    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]

        # delete both nodes from list
        nodes = nodes[:-2]

        # create tree with both nodes as children
        node = NodeTree(key1, key2)

        # add parent node to nodes
        nodes.append((node, c1 + c2))

        # sort nodes according to weight
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

    # get tree as root_node (pointing to all other nodes)
    root_node = nodes[0][0]
    # get dictionary representing characters with huffman binary representation
    huffman_code = huffman_code_tree(root_node)

    # for (key, value) in huffman_code.items():
    #     print(key, " -----> ", value)

    # create binary representation
    ans = ''
    for c in arr:
        ans += huffman_code[c]

    return ans, root_node


if __name__ == "__main__":
    ans, root = huffman_coding([1, 1, 2, 2, 3, 3, 1, 3, 3, 4, 5, 1, 1, 1, 1])

    print(ans)
    print(root)

