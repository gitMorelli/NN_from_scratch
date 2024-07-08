#include functions.h
int main(int argc, char *argv[])
{
char tsname[15]; // training set name
    srand(time(NULL));
    strcpy(tsname, "test.dat"); // default name
    if (argc > 1)
        strcpy(tsname, argv[1]); // specified name
    nex = load_ts(tsname);
    display_ts();
    bp_define_net(35, 10, 3); // inp, hid, out
    bp_reset_weights();
    bp_set_learning_rate(0.2);
    bp_set_momentum(0.5);
    interpreter();
    return 0;
}

void interpreter()
{
    char c;
    do {
        c = getchar();
        switch (c) {
            case 'R': bp_reset_weights(); break;
            case '1': bp_learn_online1(0.01, 1000); break;
            case '2': bp_learn_online2(0.01, 1000); break;
            case '3': bp_learn_online3(0.01, 1000); break;
            case 'L': bp_learn_batch(0.01, 1000); break;
            default: break;
        }
    } while (c != 'x');
}
