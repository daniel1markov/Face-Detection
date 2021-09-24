# Step 1 : Run the "ImageTaking.py" Program.
# Step 2 : Run the "ModelTrainer.py" Program after you done with step 1.
# Step 3: Run this Program once you finished with step 2.
# Enjoy!

from FaceRecognition import *

# Starting a thread that importing the names from the images and then goes to the main program.
thread_0 = threading.Thread(target=Add_Names)
thread_0.start()

