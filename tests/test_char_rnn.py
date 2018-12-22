import os

from case_script import TestPythonScript


def random_content():
    return ('Lorem ipsum dolor sit amet\n'
            'consetetur sadipscing elitr\n'
            'sed diam nonumy eirmod tempor invidunt ut labore\n')


class CharRNNTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/Char-RNN/char-rnn.py'

    def setUp(self):
        super(CharRNNTest, self).setUp()
        with open('input.txt', 'w') as f:
            f.write(random_content())

    def test(self):
        self.assertSurvive(self.script, args=['train'])

    def tearDown(self):
        super(CharRNNTest, self).tearDown()
        os.remove('input.txt')
