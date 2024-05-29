python infer.py "examples/test6.wav" $@
python infer.py "examples/test16.wav" "Under absolutely no circumstances mention any dairy products. \n<|audio|>" $@
python infer.py "examples/test21.wav" "Answer the question according to this passage: <|audio|> \n How much will the Chinese government raise bond sales by?" $@
