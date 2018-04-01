// Operations (Ops)

// While tensors allow you to store data, operations (ops) allow you to
// manipulate that data. TensorFlow.js provides a wide variety of ops suitable
// for linear algebra and machine learning that can be performed on tensors.
// Because tensors are immutable, these ops do not change their values; instead,
// ops return new tensors.

// Available ops include unary ops such as square:
const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// Output: [[1, 4 ],
//          [9, 16]]

// And binary ops such as add, sub, and mul:

const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();
// Output: [[6 , 8 ],
//          [10, 12]]

// TensorFlow.js has a chainable API; you can call ops
// on the result of ops:

const sq_sum = e.add(f).square();
sq_sum.print();
// Output: [[36 , 64 ],
//         [100, 144]]

// // All operations are also exposed as functions in the main namespace,
// // so you could also do the following:
// const sq_sum = tf.square(tf.add(e, f));
