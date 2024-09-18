from gbigsmiles import camel_to_snake, snake_to_camel


def test_camel_to_snake():
    assert camel_to_snake("camelCaseString") == "camel_case_string"
    assert camel_to_snake("AnotherExampleHere") == "another_example_here"
    assert camel_to_snake("SimpleTest") == "simple_test"
    assert camel_to_snake("AlreadySnakeCase") == "already_snake_case"
    assert camel_to_snake("XMLHttpRequest") == "xml_http_request"
    assert camel_to_snake("FahrenheitToCelsius") == "fahrenheit_to_celsius"


def test_snake_to_camel():
    assert snake_to_camel("snake_case_string") == "SnakeCaseString"
    assert snake_to_camel("another_example_here") == "AnotherExampleHere"
    assert snake_to_camel("simple_test") == "SimpleTest"
    assert snake_to_camel("alreadyCamelCase") == "AlreadyCamelCase"
    assert snake_to_camel("user_name") == "UserName"
    assert snake_to_camel("multiple_words_test") == "MultipleWordsTest"
