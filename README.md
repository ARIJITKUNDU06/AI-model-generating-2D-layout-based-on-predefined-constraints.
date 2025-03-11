The project is build on the task Circuit board component placement (Input: board size, power constraints → Output: efficient component layout).
Instructions to run this project:
Save the folder. Train the model , after training the model layout_model.h5 file will be generated. Connect the h5 file and the model with the backend for this project django has been used. Create the django account.
Open the terminal . Run the command cd myproject. Check whether the manage.py is visible or not. manage.py is available in the myproject.
Run the command python manage.py runserver under myproject command.
After that the localhost link will be generated .With the host id add this /users/predict
Then the prediction will be generated.
Next to view the image with the host id add /users/get_layout_image
