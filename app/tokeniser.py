text = "who is the PM of india?"
import tiktoken

tokenizer = tiktoken.encoding_for_model(model_name="gpt-4")

tokenId = tokenizer.encode(text)

print(tokenId)

resultIds = [47, 76, 315, 6890, 374, 35489, 41, 72, 13]

result = tokenizer.decode(resultIds)
print(result)