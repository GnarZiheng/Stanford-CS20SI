## Tensorflow Manage Experiments

###### 1. Visualize graphs with TensorBoard

~~~python
# define model
# launch a session to compute the graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs", sess.graph)       #
    for step in range(training_steps):
        sess.run([optimizer])
# Go to terminal, run:
# $ python [yourprogram].py
# $ tensorboard --logdir="./graphs" --port 6006
# Then open your browser and go to: http://localhost:6006/
~~~



###### 2. Saving and Restoring Variables

~~~Python
# checkpoint
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")  # 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                   global_step = global_step)   # 
# define model
# launch a session to compute the graph
saver = tf.train.Saver()    # 
with tf.Session() as sess:
    for step in range(training_steps):
        sess.run([train_op])
        if (i + 1)% 300 == 0:  # 
            saver.save(sess, './checkpoints/ckpt', global_step=global_step)  # 
~~~

~~~python
# define model
# launch a session to compute the graph
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/ckpt'))   # 
    if ckpt and ckpt.model_checkpoint_path:    # 
        saver.restore(sess, ckpt.model_checkpoint_path)   # 
    for step in range(training_steps):
        sess.run([optimizer])   
~~~



###### 3. Visualize our summary statistics during our training

~~~Python
# define model
with tf.name_scope("summaries"):   #      
    tf.summary.image('input', x_image, 4) #  
    tf.summary.scalar("accuracy", accuracy)  #  
    tf.summary.histogram("loss", cross_entropy)#   
    summary_op = tf.summary.merge_all()  #  

# launch a session to compute the graph
with tf.Session() as sess:    
    writer = tf.summary.FileWriter("./graphs", sess.graph)    #   
    for step in range(training_steps):
        sess.run([train_op])
        summary = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], 
                                                  keep_prob: 0.5})       #   
        writer.add_summary(summary, global_step = i)     #   
~~~



###### 4. Question about Session

**`sess.run(train_op)` vs `op.run()` vs `op.eval()`**

- If you have a `Tensor` op, calling `op.eval()` or `op.run()`is equivalent to calling `tf.get_default_session().run(t)`
- The most important difference is that you can use `sess.run()` to fetch the values of many tensors in the same step:`tf.get_default_session().run([op1,op2,op3, op4])`

~~~python
# define model
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

# launch a session to compute the graph
with tf.Session() as sess:
    for step in range(training_steps):
        _ , loss, acc, summary = sess.run([train_step, loss, acc, summary], 
                                           feed_dict = feed_dict)   # ⛳️  
    # The Optimizer operation have to run it, if you need other op value, just running it
~~~




######6. Op level seed vs Graph level seed

-		Op level seed: each op keeps its own seed		

~~~Python
## Sessions keep track of random state, Each new session restarts the random state 
c = tf.random_uniform([], -10, 10, seed=2)
with tf.Session() as sess:
     print sess.run(c) # >> 3.57493
     print sess.run(c) # >> -5.97319
   
c = tf.random_uniform([], -10, 10, seed=2)
with tf.Session() as sess:
     print sess.run(c) # >> 3.57493
with tf.Session() as sess:
     print sess.run(c) # >> 3.57493
~~~

- Graph level seed

~~~Python
tf.set_random_seed(2)
c = tf.random_uniform([], 10, 10)
d = tf.random_uniform([], 10, 10)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d)
~~~



###### 7.Data Readers



###### 8. gradients




​				
​			
​		
​	