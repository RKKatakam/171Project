// type "rcfe" and press enter to automatic create a function with the file name
import React, { useState } from "react";
//import Logo from "../assets/movie_cam.png";
import CameraRollIcon from "@mui/icons-material/CameraRoll";
import MenuIcon from "@mui/icons-material/Menu";
import {Link} from "react-router-dom";
import "../styles/Navbar.css"

function Navbar() {

    /* These consts help the menu button reveal the links to Home, Menu, etc. upon click when screen is small*/
  const [openLinks, setOpenLinks] = useState(false);

  const toggleNavbar = () => {
    setOpenLinks(!openLinks); 
  };

  return (
    <div className="navbar">
        <div className="leftSide" id={openLinks ? 'open' : 'close'}>
            <div className="camera"><CameraRollIcon /></div>
            <div className="hiddenLinks">
              <Link to="/"> Home </Link>
              <Link to="/menu"> Menu </Link> 
              <Link to="/about"> About Us</Link>
            </div>
            
        </div>
        <div className="rightSide">
            <Link to="/"> Home </Link>
            <Link to="/menu"> Menu </Link> 
            <Link to="/about"> About Us</Link>
            <button onClick={toggleNavbar}className="menu"> 
              <MenuIcon /> 
            </button>
        </div>
    </div>
  );
}

export default Navbar;
