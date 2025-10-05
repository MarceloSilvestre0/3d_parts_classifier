import os
from MathPartProcess import MathPartProcess
from dotenv import load_dotenv

load_dotenv()

output_directory = os.getenv("OUTPUT_DIRECTORY")
input_directory = os.getenv("INPUT_DIRECTORY")

stl_cather = MathPartProcess()

stl_cather.execute_render(input_directory, output_directory, views=2)