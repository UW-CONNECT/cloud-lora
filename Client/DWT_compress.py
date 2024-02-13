import gzip
import lz4.frame
import numpy
import pywt


def ret_dden(data):
    return ddencmp(numpy.real(data))


def ddencmp(data):
    coefs = pywt.wavedec(data, 'db1', level=1)
    normalizer = numpy.mean(abs(coefs[1]))
    if normalizer == 0:
        return 0.05 * max(abs(coefs[0]))
    else:
        return normalizer


def complexThresh(testSesh, coefs, multiplier, upsampling_factor=8):
    #thrR = ddencmp(numpy.real(testSesh)) * multiplier
    thrR = multiplier
    for i in range(len(coefs)):
        coefs[i] = pywt.threshold(coefs[i], thrR, 'hard')
    return


def DWT_compress(ActiveSession, multiplier):
    coefs = pywt.wavedec(ActiveSession, 'db5', level=5)
    # we perform a hard threshold and don't threshold the approximation coefficients (last level)
    # thresholding is based on the complex magnitude now
    complexThresh(ActiveSession, coefs, multiplier)

    levels = [len(x) for x in coefs]
    levels = numpy.asarray(levels, dtype=numpy.float32)

    compressThis = numpy.ndarray(0, dtype=numpy.complex64)
    for i in range(len(coefs)):
        compressThis = numpy.append(compressThis, coefs[i])
    compressThis = compressThis.view(numpy.float32)

    return (compressThis, levels)


def DWT_rec(coef, sizes):
    form = []
    index = 0
    for i in range(len(sizes)):
        level = coef[index:index + sizes[i] * 2]
        index = index + sizes[i] * 2
        insert = numpy.asarray(level, dtype=numpy.float32)
        insert = insert.view(numpy.complex64)
        form.append(insert)
    rec = pywt.waverec(form, 'db5')
    return rec


########################### gzip whole file #######################
def gzipBuffer(loc, buffer, level):
    loc += 'TX.gz'
    buffer = buffer.tobytes()
    with gzip.open(loc, 'wb', compresslevel=level) as f:
        f.write(buffer)


################### lz4 whole file ###############################
def lz4Buffer(loc, buffer, level):
    loc += 'TX.lz4'
    buffer = buffer.tobytes()
    with lz4.frame.open(loc, mode='wb', compression_level=level, block_size=lz4.frame.BLOCKSIZE_MAX64KB) as fp:
        bytes_written = fp.write(buffer)


########################### write whole file #######################
def writeBuffer(loc, buffer):
    loc += 'TX'
    buffer.tofile(loc)