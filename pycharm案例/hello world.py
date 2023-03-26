# 在Tensorflow 2执行Tensorflow 1.x版本代码
import tensorflow.compat.v1 as tf
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
# 改为图执行模式运行
tf.disable_eager_execution()
# 常量
# import tensorflow as tf

print(tf.__version__)

# 常量操作
hello = tf.constant("hello tensorflow")
# 创建会话
sess = tf.Session()

print(sess.run(hello))
sess.close()
