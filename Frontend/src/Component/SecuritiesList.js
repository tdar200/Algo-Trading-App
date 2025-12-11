import React from "react";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Divider from "@mui/material/Divider";

const STOCK_LIST = ["AMZN", "GOOG", "IBM", "TSLA", "MSFT"];

function SecuritiesList({ setStockHandler }) {
  return (
    <Box sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}>
      <nav aria-label='stock list'>
        <List>
          {STOCK_LIST.map((symbol) => (
            <ListItem key={symbol} disablePadding>
              <ListItemButton onClick={() => setStockHandler(symbol)}>
                <ListItemText primary={symbol} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </nav>
      <Divider />
    </Box>
  );
}

export default SecuritiesList;
