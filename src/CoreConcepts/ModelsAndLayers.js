// Models and Layers

// Conceptually, a model is a function that given some input will produce some
// desired output.

// In TensorFlow.js there are two ways to create models. You can use ops
// directly to represent the work the model does. For example:

// Define function
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  // More on tf.tidy in the next section
  return tf.tidy(() => {
    const x = tf.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

// Define constants: y = 2x^2 + 4x + 8
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

// Predict output for input of 2
const result = predict(2);
result.print() // Output: 24

// You can also use the high-level API tf.model to construct a model out of
// layers, which are a popular abstraction in deep learning. The following code
// constructs a tf.sequential model:

const model = tf.sequential();
model.add(
  tf.layers.simpleRNN({
    units: 20,
    recurrentInitializer: 'GlorotNormal',
    inputShape: [80, 4]
  })
);

const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({optimizer, loss: 'categoricalCrossentropy'});
model.fit({x: data, y: labels)});

// There are many different types of layers available in TensorFlow.js. A few
// examples include tf.layers.simpleRNN, tf.layers.gru, and tf.layers.lstm.
