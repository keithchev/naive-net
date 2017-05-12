Naive-net organization

models.py

	.Model
			.add_layer(<layer instance>)
			.loss(<loss instance>)
			.reset()
			.forward(x, y)
			.backward()
			.predict(x)


To build a model, call .add_layer to add layer instances:

layers.py

	.Dense
	.BatchNorm
	.Activation


<layer instance>
			   .parameters = ['kernel', 'bias']
			   .kernel = Parameter(name, shape, initializer, trainable)

parameter.py
		.Parameter
				.reset
				.value
				.gradient
				.optimization_data = {running_velocity:, running_moment: }

losses.py
		.SoftmaxLoss(num_classes)
								.calc_loss(out, y)
								.calc_dloss()



library of activation functions, called only by layer.Activation:

activations.py
				.get


trainer.py
			.Trainer
					.data = {x_train, y_train, x_val, y_val}
					.model = <model instance>
					.optimizer = <optimizer instance>

					.reset
					.train
					.accuracy


optimizers.py

			.VanillaSGD
			.MomentumSGD
			.Rmsprop
			.Adam
