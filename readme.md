#Rule of ML

##git
git init 
git add README.md
git commit -am "first commit"
git remote add origin https://github.com/philipdongfei/ML_TW.git 
git push -u origin master

##pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
exec "$SHELL"
pyenv install 3.6.8

rm -rf $PYENV_ROOT


