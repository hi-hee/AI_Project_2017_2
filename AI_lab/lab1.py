import tensorflow as tf
import matplotlib.pyplot as plt


x_data = [1,2,3]
y_data = [1,2,3]

#1. w의 초기값을 random으로 설정 (10~-10 사이값으로 범위를 지정)
w = tf.Variable(tf.random_normal([1],name='weight'))

#2. Cost 함수의 시각화
W=[]
COST=[]

for i in range(-3,6,1):
    W.append(i)
    tmp = 0
    for j in range(0,3):
        tmp += (i*x_data[j] - y_data[j])**2
    COST.append(tmp/3)

plt.plot(W,COST,'g-')
plt.show()

'''
#교수님 방법 (tensorflow 이용)

x = [1,2,3]
y = [1,2,3]

w = tf.placeholder(tf.float32)
out = x*w

cost = tf.reduce_mean(tf.square(out-y))

s = tf.Session()
s.run(tf.global_variables_initializer())

w_x = []
cost_y = []

for step in range(-30,50):
    w_step = step*0.1
    cost_val, w_val = s.run([cost,w],feed_dic ={W:w_step})
    w_x.append(w_val)
    cost_y.append(cost_val)


plt.plot(w_x,cost_y)
plt.show()

'''

#4-1.

x_data = [1,2,3]
y_data = [1,2,3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


out = x*w
cost = tf.reduce_mean(tf.square(out-y))
learning_rate = 0.01

gradient = tf.reduce_mean((w*x-y)*x)
descent = w - learning_rate * gradient
update = w.assign(descent)

s = tf.Session()
s.run(tf.global_variables_initializer())

for step in range(2001):
    s.run(update,feed_dict={x:x_data,y:y_data})
    if step % 20 ==0:
        print(step, s.run(cost,feed_dict={x:x_data,y:y_data}),s.run(w))


#4-2. optimizer를 이용한 방법
y = w*x_data
loss = tf.reduce_mean(tf.square(y-y_data))          #cost 함수의 미분식의 표현
optimizer = tf.train.GradientDescentOptimizer(0.01) #learning rate = 0.01
train = optimizer.minimize(loss)                     #미분값을 최소로 하도록 learning

sess = tf.Session()


#변수 초기화
init = tf.global_variables_initializer()
sess.run(init)

#5. 2000번의 학습 실행 및 20번마다 w&cost 출력
for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print('num=',step,', W=',sess.run(w),'cost=',sess.run(loss))

sess.close()




    #plt.plot(x_data,y_data,'bo')
    #plt.plot(x_data,sess.run(w)*x_data,'r')
    #plt.legend()
    #plt.show()




