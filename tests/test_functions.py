from pylabel import functions

def test_haversine():
    assert int(functions.haversine(52.370216, 4.895168, 52.520008,
    13.404954)) == int(946.3876221719836)