const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const fs = require('fs');

// Load an image from the file system
const image = fs.readFileSync('./image.jpg');

// Create a tensor from the image
const imageTensor = tf.node.decodeImage(image);

// Load the pre-trained model
mobilenet.load().then(model => {
    // Classify the image
    model.classify(imageTensor).then(predictions => {
        console.log(predictions);
    });
});
