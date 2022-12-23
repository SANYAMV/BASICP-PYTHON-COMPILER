import operator
import sys
ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '>': operator.gt,
    '>=': operator.ge,
    '!=': operator.ne,
    'not': operator.not_,
    '&': operator.and_,
    'and': operator.and_,
    'or': operator.or_,
}  # creating a set from which we could use operator function later on , whenever required

# DEFINING A FUNCTION THAT WOULD RETURN THE 1ST INDEXED ELEMENT OF A TUPLE PRESENT IN THE LIST(IF ANY)


def finding_tuple(a, e1):
    # INPUT : 'a' is a list with mixed elements
    #      : e1 is the 0th indexed element of the tuple for which we are checking in the list
    # OUTPUT: if the given element e1 is present in the list in the form of a tuple as suppose(e1,B) then this function would give B as its output else -1
    n = len(a)
    for i in range(0, n):
        # USING isinstance(in-built) function to check if the given element of the list if tuple or not
        if isinstance(a[i], tuple):
            '''
            isinstance is a function that takes two arguments,
            one is an element of and other is type for which we are checking.
            if the given element is the  of the type mentioned as argument then the function would return True, and False otherwise.'''
            (A, B) = a[i]  # if the function comes into this loop then surely the element is a tuple and we can express the tuple in the form (A,B)
            # BASICALLY WE ARE TRYING TO GET THE B SO WE CAN CALCULATE THE VALUE OF VARIABLE , WHICH CAN BE USED IN THE NEXT PART OF ASSIGNMENT.
            if A == e1:  # checking if the 0th indexed element of the tuple matches the given element e1
                return B  # if it does match then returning B
    return -1  # else returning -1

# DEFINING A FUNCTION THAT WOULD RETURN THE POSITION OF THE TUPLE IN THE LIST


def finding_tuple_index(a, e1):
    # INPUT : 'a' is a list with mixed elements
    #      : e1 is the 0th indexed element of the tuple for which we are checking in the list
    # OUTPUT: is the index of the tuple which has e1 as its zeroth indexed position, if it is present, else it would return -1
    n = len(a)
    for i in range(0, n):
        # USING isinstance(in-built) function to check if the given element of the list if tuple or not
        if isinstance(a[i], tuple):
            '''
            isinstance is a function that takes two arguments,
            one is an element of and other is type for which we are checking.
            if the given element is the  of the type mentioned as argument then the function would return True, and False otherwise.'''
            (A, B) = a[i]  # if the function comes into this loop then surely the element is a tuple and we can express the tuple in the form (A,B)
            # BASICALLY WE ARE TRYING TO GET THE i , WHICH CAN BE USED IN THE NEXT PART OF ASSIGNMENT.
            if A == e1:  # checking if the 0th indexed element of the tuple matches the given element e1
                return i  # if it does match then returning i
    return -1  # else returning -1

# Defining a function that will check if a given element is present in the given set


def member(s, e):
    # INPUT:s is the given set
    # INPUT:e is the element which is to be checked whether its already present in the given list
    # OUTPUT:this function returns True if e is present in the list s and False if not
    n = len(s)
    Found = False  # Defining a variable that would be our answer(bool)
    for i in range(0, n):  # a loop that will check each element of the given set
        # This loop will run for n times
        if s[i] == e:  # as soon as the loop detects that an element of the set is equal to e, it changes the value of Found
            Found = True
    return Found  # returning bool

# DEFINING A COMPLIMENTARY FUNCTION THAT WOULD HELP IN FINDING THE INDEX OF THE VALUE STORED IN THE DATA LIST FOR A VARIABLE IN THE FOLLOWING PART OF THE ASSIGNMENT


def index(a, e):
    # INPUT :'a' is a list with mixed elements
    #      : e is an integer or the value for which we are finding the index at which it is present in the list 'a'
    # OUTPU : the index of the element e in the list 'a'
    n = len(a)
    found = False  # this variable will help us finding if the given element e is present in the list 'a'
    for i in range(0, n):
        if a[i] == e:  # if e is present in the list then change the variable found to True
            found = True  # Updating the value of found
            return i  # returning the index of the element e present in the list 'a'
        # if the function exits this loop, then the element e is not present in the lis
    '''
    so what we will do is appending the element to the last of the list, and in the following part of assignment the list would be nothing but DATA list
    and hence increasing its length by one (len(DATA) is now n+1) and since the value e is appended at the last of the list then its index would be [len(DATA)-1]
    and initially len(DATA) was n and now n+1 so its index should be n only  '''
    if found == False:  # checking if found == False
        DATA.append(e)  # if it is then appending it to the last of the list
        return n  # returning the index as n

# DEFINING A FUNCTION THAT WOULD RETURN THE GARBAGE LIST IN THE DATA MEMEORY


def garbageList(a):
    n = len(a)
    l = []  # A LIST WHICH WILL HELP US IN STROINF THE INDICES OF THE ELEMENTS IN DATA LIST WITH ARE CURRENTLY REFFERED TO OR THOSE VALUES WHICH ARE NOT GARBAGE
    garbage = []  # OUR GARBAGE LIST
    for i in range(0, n):
        '''
        SO WE WILL BE LOOKING AT EACH ELEMENT IN THE LIST IF IT IS A TUPLE THAT MEANS A VARIABLE IS THEIR INSIDE THE TUPLE WHICH HAS THE INDEX FOR THE VALUE OF 
        THE VARIABLE AND THEN WE WILL BE STORING THE INDEX OF THAT TUPLE IN A LIST AND THE INDEX INSIDE THE TUPLE IN THE SAME LIST'''
        '''
        THEN WE WILL BE APPENDING ALL THE VALUES AT THE INDEX OTHER THAN THOSE STORED INSIDE THE LIST '''
        if isinstance(a[i], tuple):
            (A, B) = a[i]
            l.append(i)  # APPENDING THE INDEX OF THE TUPLE
            l.append(B)  # APPENDING THE INDEX INSIDE THE TUPLE
    for i in range(0, n):
        if member(l, i):  # IF WE COME ACROSS AN ELEMENT WHICH IS ALREADY A PART OF LIST l THEN WE HAVE TO CONTINUE
            continue
        else:  # OTHERWISE APPEND THOSE VALUES
            garbage.append(DATA[i])
    print("THE GARBAGE VALUES IN DATA MEMORY:", garbage)

# DEFINING A FUNCTION THAT WOULD RETURN THE NAMES AND THE VALUES OF EACH VARIABLE


def nameANDvalue(a):
    n = len(a)
    '''
    SINCE WE HAVE TO PRINT THE VALUES OF VARIABLES ONLY, SO WILL WE ONE BY ONE CHECK IF AN ELEMENT IS A TUPLE AND THEN FIND THE VARIABLE AND ITS VALUE 
    USING THE INDEX INSIDE THE TUPLE AND WILL RETURN BOTH'''
    for i in range(0, n):
        if isinstance(a[i], tuple):  # WE HAVE TO DEAL ONLY WITH TUPLES
            # A IS THE VARIABLE AND B IS THE INDEX AND HENCE a[B] is THE VALUE OF VARIABLE
            (A, B) = a[i]
            print("NAME OF THE VARIABLE:", A, "AND ITS VALUE IS ", a[B])


lines = []  # INTRODUCING AN EMPTY LIST THAT WOULD STORE ALL THE LINES FROM THE demofile.txt
# using the open function for opening the demofile.txt and read it.
with open('a.txt') as f:
    # using readines function to read every single element of  various lines
    lines = f.readlines()
DATA = []  # EMPTY LIST THAT WOULD STORE ALL THE DATA USED IN THE ASSIGNMENT
a = []  # INTRODUCING A LIST THAT WOULD STORE ALL THE VARIABLES OF THE demofile.txt
'''
if demofile contains: 
x = 1 + 3
y = 4
x = 5
then lines would be 'x = 1 + 3' & 'y = 4' & 'x = 5'.
statements would be x = 1 + 3 (statement number 1)
                    y = 4     (statement number 2)
                    x = 5     (statement number 3)
tokens_list for statement number 1 would be ['x' ,'=', '1' ,'+', '3']
and so on

'''
for statements in lines:  # a loop for making a list that will contain all the variables and will help us in the following part
    tokens_list = statements.split()  # splitting each line into tokens
    # considering only zeroth indexed element
    # if current variable is already present in the list then continue
    if member(a, tokens_list[0]):
        continue
    else:
        a.append(tokens_list[0])  # else append the variable to list a
for statements in lines:  # running a loop for all the lines

    # using split function to split various terms in a given line
    tokens_list = statements.split()

    if len(tokens_list) == 5:  # for tackling all the lines which are of the type A = B + C
        # tokens_list[3] will always be a operator
        # tokens_list[1] will always be a '='
        # using ops array to declare a variable that would be helping us to operate the value of a given variable
        op = ops[tokens_list[3]]
        for tokens in tokens_list:
            '''
            if a tokens_list is ['x' ,'=', '1' ,'+', '3']
            and tokens of this tokens_list are 'x' & '=' & '1' & '+' & '3' '''
            if tokens == tokens_list[1]:  # tokens_list[1] will always be a '='
                continue  # we have nothing to do with this token
            # tokens_list[3] will always be a operator
            elif tokens == tokens_list[3]:
                continue  # already taken care of we here also no need to do anything
            # considering the tokens_list[2] element
            elif tokens == tokens_list[2]:
                # now this element can be a  TERM, TERM BINARY OPERATOR TERM
                try:
                    if member(a, tokens_list[2]):  # tokens_list[2] is a variable
                        # now if it is a variable then we need to extract its value from the DATA set
                        j = finding_tuple(DATA, tokens_list[2])
                        # in this case since tokens_list[2] is a variable hence it was already stored in the DATA list so no need of appending it
                        '''if the variable is x and it is stored in the DATA set as (x,i)[HERE i IS THE INDEX OF THE ELEMENT WHICH IS PRESENT IN THE 
                        DATA LIST WHICH EQUALS THE VALUE OF VARIABLE WE ARE LOOKING FOR] then we need i and for this we have already designed a function finding_tuple '''
                        tokens_list[2] = DATA[j]  # if j is the index of the value the DATA[j] is the value of the variable which is required.
                        # converting the type of the value from string to integer.
                        tokens_list[2] = int(tokens_list[2])
                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[2] == "True":
                        # converting the string type True to bool type True
                        tokens_list[2] = True
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[2]):
                            continue
                        else:  # if not present then appending it to the list
                            DATA.append(tokens_list[2])
                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[2] == "False":
                        # converting the string type False to bool type False
                        tokens_list[2] = False
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[2]):
                            continue
                        else:
                            # if not present then appending it to the list
                            DATA.append(tokens_list[2])

                    # tokens_list[2] is anything other than a variable and a boolean value
                    elif type(tokens_list[2]) == str:
                        tokens_list[2] = int(tokens_list[2])
                        # checking if already present in the list DATA.
                        if member(DATA, int(tokens)):
                            continue
                        else:  # if not present
                            # then append the value to DATA list.
                            DATA.append(int(tokens))
                except ValueError:  # PRINTING ERROR MESSAGE
                    print("Value", tokens_list[2], "is not defined")
                    sys.exit()  # exiting the program
            # considering the tokens_list[4] element
            elif tokens == tokens_list[4]:
                try:
                    if member(a, tokens_list[4]):
                        # now this element can be a  TERM, UNARY OPERATOR TERM , TERM BINARY OPERATOR TERM
                        q = finding_tuple(DATA, tokens_list[4])
                        # in this case since tokens_list[4] is a variable hence it was already stored in the DATA list so no need of appending it
                        '''if the variable is x and it is stored in the DATA set as (x,i)[HERE I IS THE INDEX OF THE ELEMENT WHICH IS PRESENT IN THE 
                        DATA LIST WHICH EQUALS THE VALUE OF VARIABLE WE ARE LOOKING FOR] then we need i and for this we have already designed a function finding_tuple '''
                        tokens_list[4] = DATA[q]  # if q is the index of the value the DATA[q] is the value of the variable which is required.
                        # converting the type of the value from string to integer.
                        tokens_list[4] = int(tokens_list[4])
                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[4] == "True":
                        # converting the string type True to bool type True
                        tokens_list[4] = True
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[4]):
                            continue
                        else:
                            # if not present then appending it to the list
                            DATA.append(tokens_list[4])
                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[4] == "False":
                        # converting the string type False to bool type False
                        tokens_list[4] = False

                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[4]):
                            continue
                        else:
                            # if not present then appending it to the list
                            DATA.append(tokens_list[4])

                    # tokens_list[4] is anything other than a variable
                    elif type(tokens_list[4]) == str:
                        tokens_list[4] = int(tokens_list[4])
                        # checking if already present in the list DATA
                        if member(DATA, int(tokens)):
                            continue
                        else:  # if not present
                            # then append the value to DATA list.
                            DATA.append(int(tokens))
                except ValueError:
                    # PRINTING ERROR MESSAGE
                    print("value", str(tokens_list[4]), "is not defined")
                    sys.exit()  # exiting the program
        # now we will be finding the value of the variable
        '''now we will be using the values of tokens_list[2] and tokens_list[4] and calculate the value tokens_list[0] that is our variable.
        if its value is already present in the list then we will update the value of i in (variable,i)
        else appending the tuple (variable,j) as our value of the variable.'''
        val_new = op(tokens_list[2], tokens_list[4])
        i = index(DATA, val_new)
        # if val_new was already present in the DATA list and then we find its index then our function defined above will return the index .
        # if it was not present then our function index will append it to the last of the DATA list and then will return the last index and hence job done!
        if finding_tuple_index(DATA, tokens_list[0]) == -1:

            # if the value of the variable was not present then it will append the tuple (variable, index) in the DATA list
            DATA.append((tokens_list[0], i))
        else:
            # finding the index of the DATA list at which the tuple was present so that we can update the value of i
            j = finding_tuple_index(DATA, tokens_list[0])

            # updating the value of index in tuple
            DATA[j] = (tokens_list[0], i)
            '''
            in this part we tackled the case when tuple containing the value of the variable was already present and then we need to update the value of
            i '''
    if len(tokens_list) == 4:

        for tokens in tokens_list:
            if tokens == tokens_list[1]:  # tokens_list[1] will always be a '='
                continue  # we have nothing to do with this token
            elif tokens == tokens_list[3]:
                # now this element can be a  TERM, TERM BINARY OPERATOR TERM
                try:
                    if member(a, tokens_list[3]):  # tokens_list[2] is a variable
                        # now if it is a variable then we need to extract its value from the DATA set
                        j = finding_tuple(DATA, tokens_list[3])
                        # in this case since tokens_list[2] is a variable hence it was already stored in the DATA list so no need of appending it
                        '''if the variable is x and it is stored in the DATA set as (x,i)[HERE i IS THE INDEX OF THE ELEMENT WHICH IS PRESENT IN THE 
                        DATA LIST WHICH EQUALS THE VALUE OF VARIABLE WE ARE LOOKING FOR] then we need i and for this we have already designed a function finding_tuple '''
                        tokens_list[3] = DATA[j]  # if j is the index of the value the DATA[j] is the value of the variable which is required.
                        # converting the type of the value from string to integer.
                        tokens_list[3] = int(tokens_list[3])
                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[3] == "True":
                        # converting the string type True to bool type True
                        tokens_list[3] = True
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[3]):
                            continue
                        else:
                            # if not present then appending it to the list
                            DATA.append(tokens_list[3])

                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[3] == "False":
                        # converting the string type False to bool type False
                        tokens_list[3] = False
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[3]):
                            continue
                        else:
                            # if not present then appending it to the list
                            DATA.append(tokens_list[3])
                    # tokens_list[4] is anything other than a variable
                    elif type(tokens_list[3]) == str:
                        # checking if already present in the list DATA
                        if member(DATA, int(tokens_list[3])):
                            continue
                        else:  # if not present
                            # then append the value to DATA list.
                            DATA.append(int(tokens_list[3]))
                except ValueError:  # PRINTING ERROR MESSAGE
                    print("value", tokens_list[3], "is not defined")
                    sys.exit()  # exiting the program
        '''IN THE FOLLOWING PART WE WILL BE TACKLING THE CASE WHEN WE HAVE A UNARY OPERATOR LIKE '-' OR not'''
        if tokens_list[2] == "-":  # checking if the operator is '-'
            # concatinating the '-' and term
            val_t = str(tokens_list[2])+str(tokens_list[3])
            # converting the concatenated string into integer value
            val_new_t = int(val_t)

            if member(DATA, val_new_t):  # checking if val_new_t is already present in the list DATA
                continue
            else:
                # if not present then appending i to the list DATA
                DATA.append(val_new_t)
        else:  # this is the case when UNARY operator is 'not' operator
            # formulating the value of variable and storing as val_new_t
            val_new_t = operator.not_(tokens_list[3])
            if member(DATA, val_new_t):  # checking if val_new_t is already present in the list DATA
                continue
            else:
                # if not present then appending i to the list DATA
                DATA.append(val_new_t)

        # finding the index of the value in the DATA list
        i = index(DATA, val_new_t)
        if finding_tuple_index(DATA, tokens_list[0]) == -1:
            # if the value of the variable was not present then it will append the tuple (variable, index) in the DATA list
            DATA.append((tokens_list[0], i))
        else:
            j = finding_tuple_index(DATA, tokens_list[0])
            # updating the value of index in tuple
            DATA[j] = (tokens_list[0], i)
            '''
            in this part we tackled the case when tuple containing the value of the variable was already present and then we need to update the value of
            i '''

    # for tackling the statements of type x = variable or x = integer_constant
    elif len(tokens_list) == 3:
        for tokens in tokens_list:
            '''
            if a tokens_list is ['x' ,'=', '1' ,'+', '3']
            and tokens of this tokens_list are 'x' & '=' & '1' & '+' & '3' '''
            if tokens == tokens_list[1]:  # tokens_list[1] will always be a '='
                continue  # we have nothing to do with this token
            # considering the tokens_list[2] element
            elif tokens == tokens_list[2]:
                # now this element can be a  TERM, UNARY OPERATOR TERM , TERM BINARY OPERATOR TERM
                try:
                    if member(a, tokens_list[2]):
                        i = finding_tuple(DATA, tokens_list[2])
                        # in this case since tokens_list[2] is a variable hence it was already stored in the DATA list so no need of appending it
                        '''if the variable is x and it is stored in the DATA set as (x,i)[HERE i IS THE INDEX OF THE ELEMENT WHICH IS PRESENT IN THE 
                        DATA LIST WHICH EQUALS THE VALUE OF VARIABLE WE ARE LOOKING FOR] then we need i and for this we have already designed a function finding_tuple '''
                        tokens_list[2] = DATA[i]  # if i is the index of the value the DATA[i] is the value of the variable which is required.
                        # converting the type of the value from string to integer.
                        tokens_list[2] = int(tokens_list[2])
                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[2] == "True":
                        # converting the string type True to bool type True
                        tokens_list[2] = True
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[2]):
                            continue
                        else:
                            DATA.append(tokens_list[2])  # if not present

                    # this condition will help us tackle the case when input is of boolean type
                    elif tokens_list[2] == "False":
                        # converting the string type False to bool type False
                        tokens_list[2] = False
                        # checking if the current bool is present in the DATA set or not
                        if member(DATA, tokens_list[2]):
                            continue
                        else:
                            DATA.append(tokens_list[2])  # if not present
                    # tokens_list[2] is anything other than a variable
                    elif type(tokens_list[2]) == str:
                        # checking if already present in the list DATA.
                        if member(DATA, int(tokens)) == True:
                            continue
                        else:  # if not present
                            # then append the value to DATA list.
                            DATA.append(int(tokens_list[2]))
                except ValueError:  # PRINTING ERROR MESSAGE
                    print("variable", tokens_list[2], "is not defined")
                    sys.exit()  # exiting the program

        # now we will be using the value of the tokens_list[2] to assign the value to the variable
        # introducing a new variable to assign the value of the variable
        val_tokens_list = int(tokens_list[2])
        # finding the index of the val_tokens_list in the DATA list and
        i = index(DATA, (val_tokens_list))
        # checking if tokens_list[0] or the variable was present inside DATA list as zeroth indexed element
        # if it was not present then appending the tuple (x,i)
        if finding_tuple_index(DATA, tokens_list[0]) == -1:
            DATA.append((tokens_list[0], i))
        else:  # if it was present then updating the value of index i inside the tuple
            # finding at which index it was present
            j = finding_tuple_index(DATA, tokens_list[0])
            # then replacing the tuple with a new index i
            DATA[j] = (tokens_list[0], i)
(nameANDvalue(DATA))
(garbageList(DATA))
print(DATA)
