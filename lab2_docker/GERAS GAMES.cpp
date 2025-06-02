#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <queue>

using namespace std;
void gotoxy(int x, int y)
{
    COORD coord;
    coord.X = x;
    coord.Y = y;
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
}
pair<int, int> pp(char g[100][100], int blogx, int blogy){

    queue<int> x;
    queue<int> y;
    y.push(blogy);
    x.push(blogx);
     int px[100][100];
    int py[100][100];
int lx,ly;
int a,b;
int bjo=1;

while(x.size()>0)
{
        a=x.front();b=y.front();
        x.pop();y.pop();
        if(g[a+1][b]=='$'){px[a+1][b]=a;py[a+1][b]=b;lx=a+1;ly=b;break;}
        if(g[a-1][b]=='$'){px[a-1][b]=a;py[a-1][b]=b;lx=a-1;ly=b;break;}
        if(g[a][b+1]=='$'){px[a][b+1]=a;py[a][b+1]=b;lx=a;ly=b+1;break;}
        if(g[a][b-1]=='$'){px[a][b-1]=a;py[a][b-1]=b;lx=a;ly=b-1;break;}

        if(g[a+1][b]=='.'){x.push(a+1);y.push(b);g[a+1][b]='1';px[a+1][b]=a;py[a+1][b]=b;}
        if(g[a-1][b]=='.'){x.push(a-1);y.push(b);g[a-1][b]='2';px[a-1][b]=a;py[a-1][b]=b;}
        if(g[a][b+1]=='.'){x.push(a);y.push(b+1);g[a][b+1]='3';px[a][b+1]=a;py[a][b+1]=b;}
        if(g[a][b-1]=='.'){x.push(a);y.push(b-1);g[a][b-1]='4';px[a][b-1]=a;py[a][b-1]=b;}

        if(x.size()==0){bjo=2;break;
    gotoxy(5,1);
cout<<"TURETU BUT 00";}

}


a=lx;b=ly;
int a1;int b1;

while(bjo==1)
{

    a1=px[a][b];
    b1=py[a][b];
    lx=a;ly=b;
    a=a1;b=b1;
    if(g[a][b]=='T')break;

}
if(bjo==2)return make_pair(0,0);
else return make_pair(lx, ly);
}
class player{
public:
    int x,y;
};
int main()
{

    int kpy,kpx;
    player p;
    pradzia:
	int dydis=100;
	int zodis;
    int rezultatas=0;
	char z;
	char map2[100][100];
	char maps[100][100];
	int ox,oy;
int pej,pejimai,eejimai;


{///RASO EJIMO SKAICIU
cout<<"RASYK SAVO EJIMU SKAICIU"<<endl;
cin>>pejimai;
pej=pejimai;
cout<<"RASYK PRIESO EJIMU SKAICIU"<<endl;
cin>>eejimai;}
{///MAPO DYDIS
        cout << "Sveiki atvyke i zaidima!!\n"<<"Pasirinkite zemelapio dydi"<<endl;
	while(dydis>99){
    cin >> dydis;
    if(dydis>99)cout<<"Dydis per didelis"<<endl;
	}
	cout << endl;}
{///pilodm mapa
	for (int i = 0; i < dydis + 2; i++) {
		for (int u = 0; u < dydis + 2; u++){
			if (i == 0||i==dydis+1) {
				maps[u][i] = '@';
				cout << '@';
			}
			else if (u == 0 || u == dydis + 1) {
				maps[u][i] = '@';
				cout << '@';
			}
			else {
				maps[u][i] = '.';
				cout << '.';
			}
		}
		cout << endl;

	}}
{ ///PLAYER POSITIONS
	cout<<"Kur padeti zaideja? (koordinates nuo 1 iki "<<dydis<<" imtinai)"<<endl;
	coordx:
	cin>>p.x;
	if(p.x<=0||p.x>=dydis+1){
        cout<<"Bloga koordinate. \nBandyk is naujo!"<<endl;
        goto coordx;
	}
    coordy:
	cin>>p.y;
	if(p.y<=0||p.y>=dydis+1){
        cout<<"Bloga koordinate. \nBandyk is naujo!"<<endl;
        goto coordy;
	}

	cout<<"Kur padeti priesa? (koordinates nuo 1 iki "<<dydis<<" imtinai)"<<endl;
	cordx:
	cin>>kpx;
	if(kpx<=0||kpx>=dydis+1){
        cout<<"Bloga koordinate. \nBandyk is naujo!"<<endl;
        goto cordx;
	}
    cordy:
	cin>>kpy;
	if(kpy<=0||kpy>=dydis+1){
        cout<<"Bloga koordinate. \nBandyk is naujo!"<<endl;
        goto cordy;
	}
	maps[p.x][p.y]='$';
	maps[kpx][kpy]='T';}
{///Naujas ekranas, piesiam zemelapi
	int x=0,y=0;
	system("CLS");
	gotoxy(0,2);
	while(y<dydis+2){
        while(x<dydis+2){
            cout<<maps[x][y];
            x++;
        }
        y++;
        x=0;
        cout<<endl;}



	gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;}


while(zodis!='e'){

            zodis='k';
            tryagain:
             gotoxy(dydis+10,dydis-3);cout<<"Likes ejimu skaicius";
             gotoxy(dydis+29,dydis-3);cout<<"   ";cout<<pej;
             gotoxy(dydis+10,dydis-2);cout<<"Rezultatas "<<rezultatas;
             gotoxy(0,dydis+5);
            zodis=getch();
            if(zodis==224){ ///MOVES
                zodis=getch();
                if(zodis==72){ ///I VIRSU
                    if(maps[p.x][p.y-1]=='@'||maps[p.x][p.y-1]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x][p.y-1]=='.'){
                    gotoxy(p.x,p.y+2);
                    cout<<'.';
                    maps[p.x][p.y]='.';
                    p.y--;
                    gotoxy(p.x,p.y+2);
                    maps[p.x][p.y]='$';
                    cout<<'$';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
                else if(zodis==75){ /// I kaire
                    if(maps[p.x-1][p.y]=='@'||maps[p.x-1][p.y]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x-1][p.y]=='.'){
                    gotoxy(p.x,p.y+2);
                    cout<<'.';
                    maps[p.x][p.y]='.';
                    p.x--;
                    gotoxy(p.x,p.y+2);
                    maps[p.x][p.y]='$';
                    cout<<'$';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;

                }
                }
                else if(zodis==77){ /// I desine
                    if(maps[p.x+1][p.y]=='@'||maps[p.x+1][p.y]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x+1][p.y]=='.'){
                    gotoxy(p.x,p.y+2);
                    cout<<'.';
                    maps[p.x][p.y]='.';
                    p.x++;
                    gotoxy(p.x,p.y+2);
                    maps[p.x][p.y]='$';
                    cout<<'$';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
                 else if(zodis==80){ /// I APACIA
                    if(maps[p.x][p.y+1]=='@'||maps[p.x][p.y+1]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x][p.y+1]=='.'){
                    gotoxy(p.x,p.y+2);
                    cout<<'.';
                    maps[p.x][p.y]='.';
                    p.y++;
                    gotoxy(p.x,p.y+2);
                    cout<<'$';
                    maps[p.x][p.y]='$';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
                else{ ///DONT UNDERTSNA
                    gotoxy(0,0);
                cout<<"I dunt understand, please try again                              "<<endl;
                goto tryagain;
            }}
            else if(zodis=='e'){ ///DONT UNDERSTAND
                break;
            }
            else if(zodis=='s'){ /// I APACIA SIENA
                    if(maps[p.x][p.y+1]=='@'||maps[p.x][p.y+1]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x][p.y+1]=='.'){

                    gotoxy(p.x,p.y+3);
                    cout<<'@';
                    maps[p.x][p.y+1]='@';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
            else if(zodis=='w'){ /// Virsu SIENA
                    if(maps[p.x][p.y-1]=='@'||maps[p.x][p.y-1]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x][p.y-1]=='.'){

                    gotoxy(p.x,p.y+1);
                    cout<<'@';
                    maps[p.x][p.y-1]='@';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
                        else if(zodis=='a'){ /// I kaire SIENA
                    if(maps[p.x-1][p.y]=='@'||maps[p.x-1][p.y]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x-1][p.y]=='.'){
                    gotoxy(p.x-1,p.y+2);
                    cout<<'@';
                    maps[p.x-1][p.y]='@';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
                        else if(zodis=='d'){ /// I Desine SIENA
                    if(maps[p.x+1][p.y]=='@'||maps[p.x+1][p.y]=='T'){
                    gotoxy(0,0);
                    cout<<"neteisingas ejimas                                        "<<endl;
                    goto tryagain;
                } else if(maps[p.x+1][p.y]=='.'){
                    gotoxy(p.x+1,p.y+2);
                    cout<<'@';
                    maps[p.x+1][p.y]='@';
                    gotoxy(0,0);
	cout<<"Vaiksciokite su rodyklytemis,spauskite e, kad iseit"<<endl;
                }
                }
else{
                gotoxy(0,0);
                cout<<"I duuuuuunt understand, please try again                              "<<endl;
                goto tryagain;
            }

if(maps[p.x+1][p.y]!='.'&&maps[p.x-1][p.y]!='.'&&maps[p.x][p.y+1]!='.'&&maps[p.x][p.y-1]!='.')pej=0;
pej--;
rezultatas++;

if(pej==0){ ///PRIESAS VAIKSTO
    for(int k=0;k<eejimai;k++){
            if(p.x==ox&&p.y==oy)break;
for(int i=0;i<dydis+2;i++){ /// MAP DUBLICATOR
    for(int k=0;k<dydis+2;k++)
{
    map2[i][k]=maps[i][k];
}
}
pair<int, int> prr=pp(map2,kpx,kpy); ox=prr.first;oy=prr.second;
gotoxy(0,1);cout<<ox<<" "<<oy<<"      ";
if(ox==0){gotoxy(0,0);cout<<"                                 ";cout<<"LABAI NEGRAZUS EJIMAS ---> YOU LOSE";break;}
gotoxy(kpx,kpy+2);
cout<<'.';
maps[kpx][kpy]='.';
gotoxy(ox,oy+2);
cout<<'T';
maps[ox][oy]='T';
kpy=oy;kpx=ox;

if(p.x==ox&&p.y==oy)break;
}
pej=pejimai;

}
if(ox==0)break;
if(p.x==ox&&p.y==oy)break;

    }


{ /// REZAs
gotoxy(0,dydis+5);
	cout<<"Jus isgyvenote "<<rezultatas<<" ejimus!";
	ifstream in("bestRez.txt");
	int bestRez;
	in>>bestRez;
	ofstream out("bestRez.txt");
	cout<<"Iki siol geriausias rezultatas buvo: "<<bestRez<<endl;
	if(rezultatas>bestRez){
        cout<<"Sveikinu pagerinus rezultas "<<rezultatas-bestRez<<" taskais"<<endl;
        out<<rezultatas;
	}
	cout<<"again?  (y/n)";
	cin>>z;
	rezultatas=0;
	if(z=='y'||z=='Y') goto pradzia;}

	return 0;
}
