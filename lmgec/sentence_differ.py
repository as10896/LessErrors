import difflib

def sentence_differ(before, after):
    d = difflib.Differ()
    before, after = before.split(), after.split()
    results = list(d.compare(before, after))
    
    output = []
    for x in results:
        if x[0] == " ":
            output.append(x[2:])
        elif x[0] == "-":
            output.append("[-%s-]" % x[2:])
        elif x[0] == "+":
            output.append("{+%s+}" % x[2:])

    return " ".join(output)


"""
In[0]:
    sentence_differ("i am look you", "I am looking forward to seeing you .")
Out[0]:
    '[-i-] {+I+} am [-look-] {+looking+} {+forward+} {+to+} {+seeing+} you {+.+}'

In[1]:
    sentence_differ("I am look you", "I am you .")
Out[1]:
    'I am [-look-] you {+.+}'

In[2]:
    sentence_differ("this is nto hte pizzza that i ordering", "This is not the pizzza that i ordering .")
Out[2]:
    '[-this-] {+This+} is [-nto-] [-hte-] {+not+} {+the+} pizzza that i ordering {+.+}'

    # Note that this function can't align the delete and the correct word, it still need to update
    # It should be the result below
    # '[-this-] {+This+} is [-nto-] {+not+} [-hte-] {+the+} pizzza that i ordering {+.+}'
"""