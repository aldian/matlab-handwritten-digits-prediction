%MNUSTURLTOMATFILE download and extract mnist files
% Convert them to  a mant file
% 

mnistRootURL = 'http://yann.lecun.com/exdb/mnist/';
mnistFiles = [
    'train-images-idx3-ubyte';
    'train-labels-idx1-ubyte';
    't10k-images-idx3-ubyte ';
    't10k-labels-idx1-ubyte '
    ];

for i=1:size(mnistFiles, 1)
    fileName = deblank(mnistFiles(i, :));
    gzName = sprintf('%s.gz', fileName);
    url = sprintf('%s%s', mnistRootURL, gzName);
    fprintf('Downloading URL: %s\n', url);
    urlwrite(url, gzName);
    gunzip(gzName);
end

[XTrain, yTrain, XTest, yTest] = loadDataFromMnistFiles();
save mnist_data.mat XTrain yTrain XTest yTest


