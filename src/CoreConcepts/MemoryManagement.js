// Memory Management: dispose and tf.tidy

// Because TensorFlow.js uses the GPU to accelerate math operations, it's
// necessary to manage GPU memory when working with tensors and variables.

// TensorFlow.js provide two functions to help with this: dispose and tf.tidy.

// dispose

// You can call dispose on a tensor or variable to purge it and free up its GPU
// memory:

const x = tf.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
const x_squared = x.square();
x.dispose();
x_squared.dispose();

// tf.tidy

// Using dispose can be cumbersome when doing a lot of tensor operations.
// TensorFlow.js provides another function, tf.tidy, that plays a similar role
// to regular scopes in JavaScript, but for GPU-backed tensors.

// tf.tidy executes a function and purges any intermediate tensors created,
// freeing up their GPU memory. It does not purge the return value of the inner
// function.

// tf.tidy takes a function to tidy up after
const average = tf.tidy(() => {
  // tf.tidy will clean up all the GPU memory used by tensors inside
  // this function, other than the tensor that is returned.
  // Even in a short sequence of operations like the one below, a number
  // of intermediate tensors get created. So it is a good practice to
  // put your math ops in a tidy!
  const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
  const z = tf.ones([4]);
  return y.sub(z).square().mean();
});
average.print() // Output: 3.5

// Using tf.tidy will help prevent memory leaks in your application. It can also
// be used to more carefully control when memory is reclaimed.

// Two important notes

// The function passed to tf.tidy should be synchronous and also not return a
// Promise. We suggest keeping code that updates the UI or makes remote requests
// outside of tf.tidy.

// tf.tidy will not clean up variables. Variables typically last through the
// entire lifecycle of a machine learning model, so TensorFlow.js doesn't clean
// them up even if they are created in a tidy; however, you can call dispose on
// them manually.
