import tensorflow as tf

def fast_gradient_sign_method(model, iImage, iEpochs=10):

	adv_image = tf.identity(iImage)
	
	#correct label
	y_correct_label = model(adv_image)

	y_shape = y_correct_label.get_shape().as_list()
	y_dim = y_shape[1]

	#binary classfier
	if y_dim == 1:
		loss_func = tf.nn.sigmoid_cross_entropy_with_logits
	else:
		loss_func = tf.nn.softmax_cross_entropy_with_logits
	
	current_loop = 0
	
	with tf.Session() as sess:
		#create a adversarial tensor
		while current_loop <= iEpochs :
			loss = loss_func(labels=y_correct_label, logits=logits)
			#compute partial derivative of x
			grad_x = tf.gradients(loss, adv_image)
			adv_image = tf.stop_gradient(adv_image + epsilon*tf.sign(grad_x))
			current_loop += 1
			
			#check wether label changes
			equal_int = tf.to_int32(tf.equal(y_correct_label,logits))
			result = tf.equal(tf.reduce_sum(equal_int),tf.reduce_sum(tf.ones_like(equal_int)))
			if not sess.run(result):
				return adv_image
	
		return adv_image



