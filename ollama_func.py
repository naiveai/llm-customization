import ollama
import json

rust_docs = {
    "with_capacity": {
        "signature": "(capacity: usize) -> Vec<T>",
        "description": """
            Constructs a new, empty Vec<T> with at least the specified capacity.

            The vector will be able to hold at least capacity elements without reallocating.
            This method is allowed to allocate for more elements than capacity. If capacity
            is zero, the vector will not allocate.

            It is important to note that although the returned vector has the minimum
            capacity specified, the vector will have a zero length. For an explanation of
            the difference between length and capacity, see Capacity and reallocation.

            If it is important to know the exact allocated capacity of a Vec, always use the
            capacity method after construction.

            For Vec<T> where T is a zero-sized type, there will be no allocation and the
            capacity will always be usize::MAX.
        """
    },
    "append": {
        "signature": "&mut self, &mut Vec<T> -> ()",
        "description": "Moves all the elements of other into self, leaving other empty."
    },
    "drain": {
        "signature": "(&mut self, range: impl RangeBounds<usize>) -> Drain<T>",
        "description": """
            Removes the subslice indicated by the given range from the vector,
            returning a double-ended iterator over the removed subslice.

            If the iterator is dropped before being fully consumed, it drops the
            remaining removed elements.

            The returned iterator keeps a mutable borrow on the vector to
            optimize its implementation.
        """
    },
    "push": {
        "signature": "(&mut self, value: T) -> ()",
        "description": "Appends an element to the back of a collection."
    },
    "pop": {
        "signature": "(&mut self) -> Option<T>",
        "description": "Removes the last element from a collection and returns it, or None if it is empty."
    },
    "pop_front": {
        "signature": "(&mut self) -> Option<T>",
        "description": "Removes the first element from a collection and returns it, or None if it is empty."
    },
    "pop_if": {
        "signature": "(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T>",
        "description": """
            Removes the last element from a collection and returns it if it satisfies the predicate,
            or None if it is empty or the predicate is not satisfied.
        """
    },
    "len": {
        "signature": "(&self) -> usize",
        "description": "Returns the number of elements in the collection."
    },
    "into_flattened": {
        "signature": "(self) -> Vec<T>",
        "description": "Takes a Vec<[T; N]> and flattens it into a Vec<T>."
    }
}

def get_function_signatures():
    return {k: v["signature"] for k, v in rust_docs.items()}

def get_function_description(func_name):
    return rust_docs.get(func_name, {}).get("description", "Function not found.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_function_signatures",
            "description": "Retrieve the signature of all relevant Vec<T> functions available to us in Rust.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_description",
            "description": "Fetch detailed documentation for a specific Vec<T> function in Rust.",
            "parameters": {
                "type": "object",
                "properties": {
                    "func_name": {
                        "type": "string",
                        "description": "The name of the Vec<T> function to get documentation for.",
                    }
                },
                "required": ["func_name"],
            }
        }
    }
]

user_query = "I want to take a 2D array and make it 1D. How could I do that?"

messages = [
    {
        "role": "system",
        "content": """
            You are a Rust documentation assistant. Respond to the user queries
            about Vec<T> in Rust exclusively using the provided functions in the tools.
            Using any other information is not allowed.

            Respond to the user with the relevant function signature and its documentation,
            modified as necessary to fit the user's query.
        """
    },
    {"role": "user", "content": user_query},
]

response = ollama.chat(
    model='llama3.1',
    messages=messages,
    tools=tools,
)

print(f"Initial response: {response}")

if response['message'].get('tool_calls'):
    messages.append(response['message'])
    
    for tool in response['message']['tool_calls']:
        func_name = tool['function']['name']
        
        if func_name == "get_function_signatures":
            result = get_function_signatures()
            output = json.dumps({"signatures": result})
        elif func_name == "get_function_description":
            func_arg = tool['function']['arguments'].get('func_name')
            description = get_function_description(func_arg)
            output = json.dumps({
                "func_name": func_arg,
                "description": description
            })
        else:
            output = json.dumps({"error": "Unknown function"})
        
        messages.append({
            "role": "tool",
            "content": output,
        })
    
    final_response = ollama.chat(
        model='llama3.1',
        messages=messages,
        tools=tools,
    )
    
    print(f"\nFinal messages: {messages}")
    print(f"\nFinal response: {final_response}")
    print(f"\nFinal answer: {final_response['message']['content']}")
else:
    print(f"\nDirect response: {response['message']['content']}")
