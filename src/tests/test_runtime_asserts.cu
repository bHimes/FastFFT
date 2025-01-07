

#include <sstream>
#include "../../include/FastFFT.h"

// FastFFT_captureStdErr must be defined so that the asserts do not call std::abort
// It is not clear yet how useful or not this is to check that runtime asserts are working, vs. hope and pray

struct FastFFTCaptureStdErr {
    // This can be an ofstream as well or any other ostream
    std::stringstream buffer;

    // Save cout's buffer here
    std::streambuf* sbuf;

    std::string test_string;

    FastFFTCaptureStdErr( ) {
        std::cerr << "CaptureStdErr\n\n";
        // Redirect cout to our stringstream buffer or any other ostream
        sbuf = std::cerr.rdbuf( );
        std::cerr.rdbuf(buffer.rdbuf( ));
    }

    ~FastFFTCaptureStdErr( ) {

        reset( );
    }

    void reset( ) {
        // When done redirect cout to its old self
        std::cout << "Log: \n";
        std::cout << buffer.str( ) << std::endl;
        std::cerr.rdbuf(sbuf);
        std::cerr << "Stderr pointer reset" << std::endl;
    }

    bool assert_message_found(std::string to_find) {
        int res = buffer.str( ).find(to_find);
        if ( res != to_find.npos )
            return true;
        else
            return false;
    }
};

int main(int argc, char** argv) {

#ifndef FastFFT_captureStdErr
    std::cout << "Error: FastFFT_captureStdErr must be defined when running this test." << std::endl;
    return 1;
#endif

    const int input_size = 64;

    FastFFT::FourierTransformer<float, float, float2, 2> FT;
    FastFFTCaptureStdErr                                 my_capture;

    FT.SetForwardFFTPlan(64, 127, 1, 64, 64, 1);
    FT.SetInverseFFTPlan(128, 128, 1, 128, 128, 1);

    bool test_passed = my_capture.assert_message_found("dimensions must be >=, y");
    my_capture.reset( );
    if ( test_passed ) {
        std::cerr << "Test passed" << std::endl;
    }
    else {
        std::cerr << "Test failed" << std::endl;
    }

    return 0;
}
