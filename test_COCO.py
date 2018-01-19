from model_coco import *
import skimage
import pylab
import skimage.io
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7, 8"

with open(vgg_path) as f:
    fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

images = tf.placeholder("float32", [1, 224, 224, 3])
tf.import_graph_def(graph_def, input_map={"images": images})

ixtoword = np.load('./data/ixtoword_COCO.npy').tolist()
n_words = len(ixtoword)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

caption_generator = Caption_Generator(
    dim_image_L=dim_image_L,
    dim_image_D=dim_image_D,
    dim_hidden=dim_hidden,
    dim_embed=dim_embed,
    batch_size=batch_size,
    n_lstm_steps=test_maxlen + 2,
    n_words=n_words,
    hard_attetion=False                )

graph = tf.get_default_graph()

# fc7_tf, generated_words_tf, alpha_list, h_list, hatt_list, out_list = caption_generator.build_generator(maxlen=test_maxlen)
fc7_tf, generated_words_tf, alpha_list = caption_generator.build_generator(maxlen=test_maxlen)

# from tensorflow.core.protobuf import saver_pb2
# saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
# saver.restore(sess, model_path)

# saver = tf.train.import_meta_graph(os.path.join(model_path, 'model-105.meta'))
# saver.restore(sess, tf.train.latest_checkpoint(model_path))

saver = tf.train.Saver()
saved_path = tf.train.latest_checkpoint(model_path)
saver.restore(sess, saved_path)

# captioning('./giraffe.jpg')


def get_attend_image(image_path, alpha_in, rank):
    src_image = Image.open(image_path)
    src_size = src_image.size
    re = alpha_in.reshape(14, 14)
    if re.max() == 0.0:
        return
    final = re / re.max() * 0.55
    cut_this = CUT * 0.7
    div0 = int(src_size[0] / 14)
    div1 = int(src_size[1] / 14)

    for i in range(0, src_size[0]):
        for j in range(0, src_size[1]):
            index0 = int(i / div0)
            index1 = int(j / div1)
            if index0 > 13:
                index0 = 13
            if index1 > 13:
                index1 = 13
            src_pixel = src_image.getpixel((i, j))
            alpha_add = final[index0, index1]
            if final[index0, index1] < cut_this:
                continue
            dst_pixel = [int(k * (1 - alpha_add) + 255 * alpha_add) for k in src_pixel]
            src_image.putpixel((i, j), (dst_pixel[0], dst_pixel[1], dst_pixel[2]))
    src_image.save(image_path[0: -3] + "-out-" + str(rank) + ".jpg")


def captioning(test_image_path=None):
    '''    
    mylist = graph.get_operations()
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(mylist)
    exit(1234)
    '''

    image_val = read_image(test_image_path)

    fc7 = sess.run(graph.get_tensor_by_name("import/conv5_3/conv5_3:0"), feed_dict={images: image_val})
    fc7 = fc7.reshape(1, 196, 512)


    # generated_word_index, alpha_out, h_out, hatt_out, out = sess.run([generated_words_tf,alpha_list,h_list,hatt_list,out_list], feed_dict={fc7_tf: fc7})
    generated_word_index, alpha_out = sess.run(
        [generated_words_tf, alpha_list], feed_dict={fc7_tf: fc7})
    generated_word_index = np.hstack(generated_word_index)


    generated_words = [ixtoword[x] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.') + 1

    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence[0:-2]
    generated_sentence = generated_sentence + '.'

    # for i in range(15):
    #     # loc = np.where(alpha_out[i] == np.max(alpha_out[i]))
    #     # alpha_out[i][loc[0][0]][loc[1][0]] = 0
    #     get_attend_image(test_image_path, alpha_out[i], i)

    print generated_sentence
    return generated_sentence


captioning('./giraffe.jpg')


