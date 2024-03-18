alias CoTEST='echo "CoTEST test"'

# Snippet : Afficher le nombre de fichiers dans le dossier courant
alias count_files='echo "Nombre de fichiers dans le dossier courant : $(ls -1A | wc -l)"'

# Snippet : Afficher la taille du dossier courant
alias folder_size='du -sh'

alias conda_activate='source ~/conda/etc/profile.d/conda.sh && conda activate'
alias mamba_activate='source ~/conda/etc/profile.d/conda.sh && source ~/conda/etc/profile.d/mamba.sh && mamba activate'


# Snippet : Afficher les informations sur un dossier (avec dossier courant par défaut)
function folder_info() {
    local folder_path="$1"

    # Utiliser le dossier courant si aucun chemin n'est spécifié
    if [ -z "$folder_path" ]; then
        folder_path="./"
    fi

    if [ -d "$folder_path" ]; then
        echo "Informations pour le dossier : $folder_path"
        echo "Nombre de fichiers : $(find "$folder_path" -maxdepth 1 -type f | wc -l)"
        echo "Nombre de dossiers : $(find "$folder_path" -maxdepth 1 -type d | wc -l)"
        echo "Taille du dossier : $(du -sh "$folder_path")"
        echo "Dernier fichier modifié : $(find "$folder_path" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' ')"
    fi
}

function condaremove(){
    conda remove -n "$1" --all
}

function condacreateenv(){
    conda create -n "$1"
}



# Snippet : update git
function git_update() {
    local update_commit="$1"
    
    # Utiliser le mot "update" si pas de message spécifié
    if [ -z "$update_commit" ]; then
        update_commit="update git"
    fi
    
    git add . && git commit -am "$update_commit" && git push
}


# arrow up
bind '"\e[A": history-search-backward'

# arrow down
bind '"\e[B": history-search-forward'



# Fonction récursive pour parcourir les fichiers et les dossiers
convert_to_pdf() {
    local folder_path="$1"
    # Utiliser le dossier courant si aucun chemin n'est spécifié
    if [ -z "$folder_path" ]; then
        echo "Il faut entrer un chemin valide !"
    fi

    if [ -d "$folder_path" ]; then
      for file in "$folder_path"/*; do
        if [ -d "$file" ]; then
          # Appel récursif si le chemin est un dossier
          convert_to_pdf "$file"
        elif [ -f "$file" ]; then
          # Conversion du fichier en PDF
          unoconv -f pdf "$file"
          
          # Suppression du fichier original
          rm "$file"
        fi
      done
    fi
}


function custom_image_list() {
# fonction pour faire un fichier texte avec une liste de fichier avec le pattern definie en 1
    local pattern="$1"
    local output_file="$2"
    
    echo "Veuillez spécifier le pattern par exemple *LRL* et le nom du fichier par exemple imFile.txt."

    find "$(pwd)" -type f -name "$pattern" | sort > "$output_file"
}


function compter_lignes_python(){
# Récupérer le répertoire courant
    repertoire_courant="$(pwd)"

# Utiliser la commande find pour rechercher tous les sous-dossiers du répertoire courant
# et exécuter une action dans chacun d'eux
    find "$repertoire_courant" -mindepth 1 -maxdepth 1 -type d | while read -r sous_dossier; do
        #echo "Sous-dossier : $sous_dossier"
        # Utilisez la commande find pour rechercher tous les fichiers .py
        # dans le répertoire courant et ses sous-dossiers
        # Puis, utilisez wc -l pour compter les lignes de chaque fichier
        total_lines=0
        for file in $(find $sous_dossier -type f -name "*.py"); do
            lines=$(wc -l < "$file")
            total_lines=$((total_lines + lines))
        done

        # Affiche le nombre total de lignes
        echo "Le nombre total de lignes de code Python dans $sous_dossier : $total_lines"
        # Vous pouvez ajouter ici l'action que vous souhaitez effectuer pour chaque sous-dossier
        # Par exemple, pour afficher le contenu du sous-dossier, vous pouvez utiliser : ls "$sous_dossier"
    done
}


function crop_images_JPG(){
    mkdir resized  # Créez un dossier pour stocker les images redimensionnées (facultatif)
    for fichier in *.JPG; do
        #convert "$fichier" -resize 400x400\! "resized/$fichier"
        convert "$fichier"  -gravity center -crop 50%x50%+0+0 "resized/$fichier"
        echo "Crop $fichier en 50 50 dans resized/$fichier"
    done
}


function resize_images_png(){
    mkdir resized  # Créez un dossier pour stocker les images redimensionnées (facultatif)
    for fichier in *.png; do
        convert "$fichier" -resize 400x400\! "resized/$fichier"
        echo "Converti $fichier en 400x400 dans resized/$fichier"
    done
}

function resize_images_jpg(){
    mkdir resized  # Créez un dossier pour stocker les images redimensionnées (facultatif)
    for fichier in *.jpg; do
        convert "$fichier" -resize 400x400\! "resized/$fichier"
        echo "Converti $fichier en 400x400 dans resized/$fichier"
    done
}



resize_images() {
    local resolution="$1"   # Récupère la résolution depuis le premier argument
    local extension="$2"    # Récupère l'extension depuis le deuxième argument

    if [ -z "$resolution" ] || [ -z "$extension" ]; then
        echo "Veuillez spécifier la résolution (par exemple, 400x400) et l'extension (par exemple, jpg) en tant qu'arguments."
        return
    fi

    mkdir -p resized  # Créez un dossier pour stocker les images redimensionnées (facultatif)

    for fichier in *."$extension"; do
        convert "$fichier" -resize "$resolution" "resized/$fichier"
        echo "Converti $fichier en $resolution dans resized/$fichier"
    done
}

# Utilisation :
# Pour redimensionner les images .jpg en 800x600, appelez la fonction comme ceci :
# resize_images "800x600" "jpg"



function remove_img_above_n()
{
    dossier="./"  # Remplacez par le chemin vers votre dossier
    extension=".jpg"  # Remplacez par l'extension de vos fichiers

# Parcourez les fichiers dans le dossier
    for fichier in "$dossier"/*"$extension"; do
        nom_fichier=$(basename "$fichier")  # Obtenez le nom du fichier sans le chemin
        nom_base=${nom_fichier%.*}  # Obtenez le nom du fichier sans l'extension
        numero_image=${nom_base#*_}  # Obtenez le numéro d'image à partir du nom

        if [ "$numero_image" -gt $1 ]; then
            rm "$fichier"  # Supprimez le fichier
            echo "Fichier supprimé : $nom_fichier"
        fi

    done
}


function skip_n_img()
{
    dossier="./"  # Remplacez par le chemin vers votre dossier
    extension=".jpg"  # Remplacez par l'extension de vos fichiers

# Parcourez les fichiers dans le dossier
    for fichier in "$dossier"/*"$extension"; do
        nom_fichier=$(basename "$fichier")  # Obtenez le nom du fichier sans le chemin
        nom_base=${nom_fichier%.*}  # Obtenez le nom du fichier sans l'extension
        numero_image=${nom_base#*_}  # Obtenez le numéro d'image à partir du nom

        #Supprimez le fichier si l'index est un multiple de 5
        if [ $((numero_image % $1)) -ne 0 ]; then
            rm "$fichier"  # Supprimez le fichier
            echo "Fichier supprimé : $nom_fichier"
        fi
    done
}

function delete_docker_image(){
    local image="$1"
    sudo docker stop $(docker ps -aq) # Pour arrêter tous les conteneurs
    sudo docker rm $(docker ps -aq)   # Pour supprimer tous les conteneurs
    sudo docker rmi "$image"
}

# fonction pour supprimer toutes les none
alias docker_delete_nones='docker images -a | grep "<none>" | awk "{print $3}" | xargs -I {} docker rmi {}'


# # fonction pour push submodules
# alias git_submodules_push='git submodule foreach "git add . && git commit -m "update" && git push"'

alias update_submodules="git submodule foreach \"if [ -n \\\"\$(git status --porcelain)\\\" ]; then git add . && git commit -m 'mise à jour' && git push; fi\""


# fonction pour pull submodules
alias git_submodules_pull='git submodule update --remote --merge'


# Fonction pour push le dépôt principal et ses sous-modules
function git_push_all() {
    local update_commit="$1"
    
    # Utiliser le mot "update" si pas de message spécifié
    if [ -z "$update_commit" ]; then
        update_commit="update git"
    fi
    
    # Ajouter, committer et pousser les modifications du dépôt principal
    git add . && git commit -am "$update_commit" && git push
    
    # # Puis, push les sous-modules
    # git submodule foreach "git add . && git commit -m 'update' && git push"

    # Parcourir tous les sous-modules
    git submodule foreach "
        # Vérifier s'il y a des modifications
        if [ -n \"\$(git status --porcelain)\" ]; then
            # S'il y a des modifications, ajouter, committer et pousser
            git add .
            git commit -m 'mise à jour'
            git push
        fi
    "
}


# Fonction pour pull le dépôt principal et ses sous-modules
function git_pull_all() {
    # Mettre à jour le dépôt principal
    git pull
    
    # Mettre à jour les sous-modules
    git submodule update --remote --merge

    # message de fin
    echo "evreything pulled"
}


# Fonction pour initialiser un nouveau sous-module
function git_init_submodule() {
    local submodule_path="$1"
    local submodule_url="$2"
    
    # Vérifier si les arguments ont été fournis
    if [ -z "$submodule_path" ] || [ -z "$submodule_url" ]; then
        echo "Usage: git_init_submodule <chemin_du_sous-module> <URL_du_sous-module>"
        return 1
    fi
    
    # Ajouter le sous-module
    git submodule add "$submodule_url" "$submodule_path"
    
    # Mettre à jour et initialiser le sous-module
    git submodule update --init --recursive
    
    # Afficher un message de confirmation
    echo "Sous-module initialisé avec succès : $submodule_path"
}



# Fonction pour initialiser un nouveau dépôt Git avec un commit initial
function git_init_commit() {
    local repo_name="$1"
    
    # Vérifier si le nom du dépôt est fourni en argument
    if [ -z "$repo_name" ]; then
        echo "Usage: git_init_commit <nom_du_dépôt>"
        return 1
    fi
    
    # Initialiser le dépôt Git
    git init
    
    # Créer un fichier README.md avec un contenu initial
    echo "# $repo_name" >> README.md
    
    # Ajouter le fichier README.md
    git add README.md
    
    # Effectuer le premier commit
    git commit -m "first commit"
    
    # Renommer la branche par défaut en 'main'
    git branch -M main
    
    # Ajouter un dépôt distant et pousser le commit initial
    git remote add origin git@github.com:sdurand777/"$repo_name".git
    git push -u origin main
    
    # Afficher un message de confirmation
    echo "Nouveau dépôt Git '$repo_name' initialisé et commit initial poussé avec succès."
}



# alias pour connnaitre les chemins include au cpp
alias gccinclude='gcc -xc++ -E -v -'

# # add torch to gcc cpp path
# PATH5="/usr/include/opencv4"
# PATH6="/home/ivm/.local/lib/python3.8/site-packages/torch/include"
# PATH7="/home/ivm/.local/lib/python3.8/site-packages/torch/include/torch/csrc/api/include"
# PATH8="/usr/include/python3.8"
# PATH9="/usr/include/eigen3"
# PATH10="/home/ivm/open3d_install/include"
# PATH11="/home/ivm/.local/include"
# export CPLUS_INCLUDE_PATH=$PATH5:$PATH6:$PATH7:$PATH8:$PATH9:$PATH10:$PATH11

# ajouter automatic login
configure_automatic_login() {
    echo "[daemon]" | sudo tee -a /etc/gdm3/custom.conf
    echo "AutomaticLoginEnable=true" | sudo tee -a /etc/gdm3/custom.conf
    echo "AutomaticLogin=$USER" | sudo tee -a /etc/gdm3/custom.conf
}

# supprimer le automatic login
remove_automatic_login() {
    if grep -q "^AutomaticLogin" /etc/gdm3/custom.conf; then
        sudo sed -i '/^AutomaticLogin/d' /etc/gdm3/custom.conf
        echo "Automatic login configuration removed."
    else
        echo "Automatic login configuration not found."
    fi
}

# ajouter un job sans log
add_reboot_cron_job() {
    if [ -z "$1" ]; then
        echo "Usage: add_reboot_cron_job ajouter script shell a lancer au reboot <script_path>"
        return 1
    fi

    SCRIPT="$1"
    (crontab -l ; echo "@reboot $SCRIPT") | crontab -
}

# retirer la mise en veille
mask_sleep() {
    sudo systemctl mask sleep.target
}

# lancer un job au reboot
add_reboot_cron_job_with_log() {
    if [ -z "$1" ]; then
        echo "Usage: add_reboot_cron_job_with_log ajouter script shell a lancer au boot <script_path>"
        return 1
    fi

    SCRIPT="$1"
    LOG_FILE="$HOME/install_log"
    (crontab -l ; echo "@reboot $SCRIPT >> $LOG_FILE 2>&1") | crontab -
}


add_user_to_docker_group() {
    sudo groupadd docker
    sudo usermod -aG docker $USER
}

clean_crontab() {
    echo "" | crontab -
}

activate_desktop_files() {
    cd "$(xdg-user-dir DESKTOP)" || return 1
    shopt -s nullglob
    FILES="*.desktop"
    for f in $FILES; do
        sudo gio set "$f" metadata::trusted true
    done
    sudo chmod +x *.desktop
}


activate_wifi() {
    # Vérifiez si les arguments sont fournis
    if [ $# -ne 2 ]; then
        echo "Usage: activate_wifi <username> <password>"
        return 1
    fi

    # Activer le WiFi
    nmcli r wifi on

    # Se connecter au WiFi en utilisant le nom d'utilisateur fourni comme SSID
    nmcli d wifi connect "$1" password "$2"
}


# update config complete
update_netplan_config_ip_and_wifi() {
    # Vérifier le nombre d'arguments
    if [ "$#" -ne 4 ]; then
        echo "Usage: update_netplan_config <addresses> <gateway> <nomreseauwifi> <password>"
        return 1
    fi

    # Récupérer le nom logique de l'interface Ethernet
    logical_name=$(sudo lshw -class network | awk '/description: Ethernet interface/,/logical name:/ {if ($1 == "logical" && $2 == "name:") print $3}')
    logical_name_wifi=$(sudo lshw -class network | awk '/description: Wireless interface/,/logical name:/ {if ($1 == "logical" && $2 == "name:") print $3}')

    echo "logical name : $logical_name"

    addresses="$1"
    gateway="$2"
    nomreseauwifi="$3"
    password="$4"

    # Vérifier si le fichier YAML existe
    yaml_file=$(find /etc/netplan/ -name "*.yaml" -type f)

    # Vérifier si un fichier YAML est trouvé
    if [ -n "$yaml_file" ]; then
        echo "Fichier YAML trouvé: $yaml_file"

        # Remplacer le contenu du fichier YAML avec le texte fourni
        echo "# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    $logical_name:
      dhcp4: no
      addresses:
        - $addresses
      gateway4: $gateway
  wifis:
    $nomreseauwifi:
      dhcp4: true
      access-points:
        \"$nomreseauwifi\":
          password: \"$password\"" > "$yaml_file"

        echo "Fichier YAML mis à jour avec les nouvelles valeurs."

        sudo netplan apply

        echo "apply new netplan"

        sudo systemctl restart systemd-networkd

        echo "restart systemd-networkd"
    else
        echo "Erreur : Aucun fichier YAML trouvé dans /etc/netplan/."
    fi
}


# update uniquement ip
update_netplan_config_ip() {
    # Vérifier le nombre d'arguments
    if [ "$#" -ne 2 ]; then
        echo "Usage: update_netplan_config <addresses> <gateway>"
        return 1
    fi

    addresses="$1"
    gateway="$2"

    # Vérifier si le fichier YAML existe
    yaml_file=$(find /etc/netplan/ -name "*.yaml" -type f)

    # Vérifier si un fichier YAML est trouvé
    if [ -n "$yaml_file" ]; then
        echo "Fichier YAML trouvé: $yaml_file"

        # Remplacer le contenu du fichier YAML avec le texte fourni
        echo "# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    $logical_name:
      dhcp4: no
      addresses:
        - $addresses
      gateway4: $gateway" > "$yaml_file"

        echo "Fichier YAML mis à jour avec les nouvelles valeurs."

        sudo netplan apply

        echo "apply new netplan"

        sudo systemctl restart systemd-networkd

        echo "restart systemd-networkd"
    else
        echo "Erreur : Aucun fichier YAML trouvé dans /etc/netplan/."
    fi
}



generate_ssh_key() {
    # Demander à l'utilisateur s'il veut générer une clé SSH
    read -p "Voulez-vous générer une clé SSH ? (oui/non): " answer

    # Vérifier la réponse de l'utilisateur
    if [ "$answer" == "oui" ]; then
        # Générer une nouvelle clé SSH
        ssh-keygen -t ed25519 -C "stan777.durand777@gmail.com"

        # Lancer l'agent SSH
        eval "$(ssh-agent -s)"

        # Ajouter la clé SSH générée à l'agent SSH
        ssh-add ~/.ssh/id_ed25519

        # Afficher la clé SSH générée
        echo "Copiez le texte suivant pour l'ajouter à votre profil GitHub :"
        cat ~/.ssh/id_ed25519.pub
    else
        echo "Génération de la clé SSH annulée."
    fi
}

# nouvelle fonction pour faire fonctionner les submodules
# Fonction pour exécuter Docker avec les options spécifiées
# Usage: run_my_docker [image_name]
#   image_name : Le nom de l'image Docker à exécuter
function run_docker_network_gpus() {
    echo "Exécute Docker avec les options spécifiées."
    echo "Usage: run_my_docker [image_name]"
    echo "  image_name : Le nom de l'image Docker à exécuter"
    local docker_image="$1"
    docker run -it --rm --network host --gpus all \
        --env=XAUTHORITY=/tmp/.docker.xauth \
        --env=DISPLAY \
        --volume=/tmp/.docker.xauth:/tmp/.docker.xauth:rw \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume=/var/run/bumblebee.socket:/var/run/bumblebee.socket:rw \
        --device /dev/dri/card0:/dev/dri/card0 \
        -v "$(pwd)":/code \
        "$docker_image"
}
