import pickle

# Check the saved maxlen
with open('maxlen_new.pkl', 'rb') as f:
    saved_maxlen = pickle.load(f)

print(f"Saved maxlen: {saved_maxlen}")