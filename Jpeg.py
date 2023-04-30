from PIL import Image
from scipy import fftpack
import numpy
from bitstream import BitStream
from numpy import *
import huffmanEncode as huffman
import time
import sys

STD_CQuantTable = numpy.array(
    [17, 18, 24, 47, 99, 99, 99, 99,
     18, 21, 26, 66, 99, 99, 99, 99,
     24, 26, 56, 99, 99, 99, 99, 99,
     47, 66, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99], dtype=int)
STD_CQuantTable = STD_CQuantTable.reshape([8, 8])
STD_LQuantTable = numpy.array(
    [16, 11, 10, 16, 24, 40, 51, 61,
     12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56,
     14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77,
     24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101,
     72, 92, 95, 98, 112, 100, 103, 99], dtype=int)
STD_LQuantTable = STD_LQuantTable.reshape([8, 8])

zigzag = numpy.array(
    [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
     28, 35, 42,
     49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62,
     63])


def main():
    src = "lena_grayscale.bmp"
    quality = 100
    outJPEG = "out" + str(quality) + ".jpg"

    numpy.set_printoptions(threshold=numpy.inf)
    srcImage = Image.open(src)
    srcWidth, srcHeight = srcImage.size
    srcImageMatrix = numpy.asarray(srcImage)

    Width = srcWidth
    Height = srcHeight
    if (srcWidth % 8 != 0):
        Width = srcWidth // 8 * 8 + 8
    if (srcHeight % 8 != 0):
        Height = srcHeight // 8 * 8 + 8

    addedMatrixMatrix = numpy.zeros((Height, Width, 3), dtype=numpy.uint8)
    for y in range(srcHeight):
        for x in range(srcWidth):
            addedMatrixMatrix[y][x] = srcImageMatrix[y][x]

    y, u, v = Image.fromarray(addedMatrixMatrix).convert('YCbCr').split()

    yMatrix = numpy.asarray(y).astype(int)
    uMatrix = numpy.asarray(u).astype(int)
    vMatrix = numpy.asarray(v).astype(int)

    yMatrix = yMatrix - 127
    uMatrix = uMatrix - 127
    vMatrix = vMatrix - 127

    if (quality <= 0):
        quality = 1
    if (quality > 100):
        quality = 100
    if (quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
    lQuantTable = numpy.array(numpy.floor((STD_LQuantTable * qualityScale + 50) / 100))
    lQuantTable[lQuantTable == 0] = 1
    lQuantTable[lQuantTable > 255] = 255
    lQuantTable = lQuantTable.reshape([8, 8]).astype(int)
    cQuantTable = numpy.array(numpy.floor((STD_CQuantTable * qualityScale + 50) / 100))
    cQuantTable[cQuantTable == 0] = 1
    cQuantTable[cQuantTable > 255] = 255
    cQuantTable = cQuantTable.reshape([8, 8]).astype(int)
    blockSum = Width // 8 * Height // 8

    yDC = numpy.zeros([blockSum], dtype=int)
    uDC = numpy.zeros([blockSum], dtype=int)
    vDC = numpy.zeros([blockSum], dtype=int)
    dyDC = numpy.zeros([blockSum], dtype=int)
    duDC = numpy.zeros([blockSum], dtype=int)
    dvDC = numpy.zeros([blockSum], dtype=int)

    BS = BitStream()

    blockNum = 0
    for y in range(0, Height, 8):
        for x in range(0, Width, 8):
            yDctMatrix = fftpack.dct(fftpack.dct(yMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            uDctMatrix = fftpack.dct(fftpack.dct(uMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            vDctMatrix = fftpack.dct(fftpack.dct(vMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T

            yQuantMatrix = numpy.rint(yDctMatrix / lQuantTable)
            uQuantMatrix = numpy.rint(uDctMatrix / cQuantTable)
            vQuantMatrix = numpy.rint(vDctMatrix / cQuantTable)

            yZCode = yQuantMatrix.reshape([64])[zigzag]
            uZCode = uQuantMatrix.reshape([64])[zigzag]
            vZCode = vQuantMatrix.reshape([64])[zigzag]
            yZCode = yZCode.astype(numpy.intc)
            uZCode = uZCode.astype(numpy.intc)
            vZCode = vZCode.astype(numpy.intc)

            yDC[blockNum] = yZCode[0]
            uDC[blockNum] = uZCode[0]
            vDC[blockNum] = vZCode[0]

            if (blockNum == 0):
                dyDC[blockNum] = yDC[blockNum]
                duDC[blockNum] = uDC[blockNum]
                dvDC[blockNum] = vDC[blockNum]
            else:
                dyDC[blockNum] = yDC[blockNum] - yDC[blockNum - 1]
                duDC[blockNum] = uDC[blockNum] - uDC[blockNum - 1]
                dvDC[blockNum] = vDC[blockNum] - vDC[blockNum - 1]

            BS.write(huffman.encodeDCT(dyDC[blockNum], 1), bool)
            huffman.encodeAC(BS, yZCode[1:], 1)
            BS.write(huffman.encodeDCT(duDC[blockNum], 0), bool)
            huffman.encodeAC(BS, uZCode[1:], 0)
            BS.write(huffman.encodeDCT(dvDC[blockNum], 0), bool)
            huffman.encodeAC(BS, vZCode[1:], 0)

            blockNum = blockNum + 1

    JPEG = open(outJPEG, 'wb+')
    JPEG.write(huffman.hexToBytes('FFD8FFE000104A46494600010100000100010000'))
    JPEG.write(huffman.hexToBytes('FFDB004300'))
    lQuantTable = lQuantTable.reshape([64])
    JPEG.write(bytes(lQuantTable.tolist()))
    JPEG.write(huffman.hexToBytes('FFDB004301'))
    cQuantTable = cQuantTable.reshape([64])
    JPEG.write(bytes(cQuantTable.tolist()))
    JPEG.write(huffman.hexToBytes('FFC0001108'))
    hHex = hex(srcHeight)[2:]
    while len(hHex) != 4:
        hHex = '0' + hHex

    JPEG.write(huffman.hexToBytes(hHex))

    wHex = hex(srcWidth)[2:]
    while len(wHex) != 4:
        wHex = '0' + wHex

    JPEG.write(huffman.hexToBytes(wHex))

    JPEG.write(huffman.hexToBytes('03011100021101031101'))

    JPEG.write(huffman.hexToBytes(
        'FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))
    sosLength = BS.__len__()
    filledNum = 8 - sosLength % 8
    if (filledNum != 0):
        BS.write(numpy.ones([filledNum]).tolist(), bool)
    JPEG.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0]))
    sosBytes = BS.read(bytes)
    for i in range(len(sosBytes)):
        JPEG.write(bytes([sosBytes[i]]))
        if (sosBytes[i] == 255):
            JPEG.write(bytes([0]))
    JPEG.write(bytes([255, 217]))
    JPEG.close()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)