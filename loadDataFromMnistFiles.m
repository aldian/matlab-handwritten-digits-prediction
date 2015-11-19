function [XTrain, yTrain, XTest, yTest] = loadDataFromMnistFiles()
    [str, maxsize, endian] = computer;
    
    fileID = fopen('train-images-idx3-ubyte', 'rb');
    [data, count] = fread(fileID, 4);
    if sum(data == [0;0;8;3]) ~= 4
        error('Invalid file signature');
    end
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nTraining = typecast(uint8(data), 'uint32');
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nRows = typecast(uint8(data), 'uint32');
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nCols = typecast(uint8(data), 'uint32');
    nBytesPerImage = nRows * nCols;
    XTrain = zeros(nTraining, nBytesPerImage);
    for i=1:nTraining
        [data, count] = fread(fileID, nBytesPerImage);
        XTrain(i, :) = data';
    end
    fclose(fileID);
    
    fileID = fopen('train-labels-idx1-ubyte', 'rb');
    [data, count] = fread(fileID, 4);
    if sum(data == [0;0;8;1]) ~= 4
        error('Invalid file signature');
    end
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nTrainLabels = typecast(uint8(data), 'uint32');
    [yTrain, count] = fread(fileID, nTrainLabels);
    fclose(fileID);
    %yTrain(find(yTrain == 0)) = 10;
    
    fileID = fopen('t10k-images-idx3-ubyte', 'rb');
    [data, count] = fread(fileID, 4);
    if sum(data == [0;0;8;3]) ~= 4
        error('Invalid file signature');
    end
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nTest = typecast(uint8(data), 'uint32');
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nRows = typecast(uint8(data), 'uint32');
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nCols = typecast(uint8(data), 'uint32');
    nBytesPerImage = nRows * nCols;
    XTest = zeros(nTest, nBytesPerImage);
    for i=1:nTest
        [data, count] = fread(fileID, nBytesPerImage);
        XTest(i, :) = data';
    end
    fclose(fileID);
    
    fileID = fopen('t10k-labels-idx1-ubyte', 'rb');
    [data, count] = fread(fileID, 4);
    if sum(data == [0;0;8;1]) ~= 4
        error('Invalid file signature');
    end
    [data, count] = fread(fileID, 4);
    if endian == 'L'
        data = flipud(data);
    end
    nTestLabels = typecast(uint8(data), 'uint32');
    [yTest, count] = fread(fileID, nTestLabels);
    fclose(fileID);
    %yTest(find(yTest == 0)) = 10;
end