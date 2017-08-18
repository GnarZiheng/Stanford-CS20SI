#### Tensorflow  separates definition of computations from their execution

##### Phase 1: assemble a graph.

- **Step 1: Read in data**

  ~~~
  # data input script
  ~~~

- **Step 2: Create placeholders for inputs and labels**

  ~~~Python
  tf.placeholder(dtype, shape=None, name=None)
  ~~~

- **Step 3:Create weight an bias**

  ~~~python
  tf.Variable(initial_value=None, trainable=True, collections=None, name=None, dtype=None,...)
  ~~~

- **Step 4: Build model to predict Y**

  ~~~
  # tensorflow operation
  ~~~

- **Step 5:Specify loss function**

  ~~~Python
  tf.nn.softmax_cross_entropy_with_logits()
  ~~~

- **Step 6: Create optimizer**

  ~~~python
  tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
  ~~~

##### Phase 2: ues a session to execute operations in the graph.

- **Step 1: Initialize variables**

  ~~~Python
  tf.global_variables_initializer()
  tf.summary.FileWriter('./graphs', sess.graph)
  ~~~

- **Step 2: Run optimizer op(with data fed into placeholders for inputs and labels) **

  ~~~python
  sess.run(op, feed_dict={X:X_batch, Y:Y_batch})
  ~~~

  ​

